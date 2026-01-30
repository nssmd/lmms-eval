"""
STAR-7B Visual Chain-of-Thought Model

Two-stage inference:
1. Stage 1: Generate visualization image using STAR's generation/editing capability
   - With original image: Uses image editing (i2i) for context-aware visualization
   - Without original image: Uses text-to-image generation
2. Stage 2: Answer question using the generated image (and optionally original image) using STAR's understanding capability

Multi-Image Support:
- Stage 1 (Generation/Editing): 
  * Image-to-image editing when original image is provided (leverages STAR's strong i2i capability)
  * Text-to-image generation when no original image is available
- Stage 2 (Understanding): Full multi-image support (original + generated images)

STAR's Image Editing Capability:
- STAR-7B excels at image editing tasks with a score of 4.34 on ImgEdit benchmark
- Supports instruction-based editing: inpainting, style transfer, object addition/removal
- Uses autoregressive token prediction conditioned on both input image and text instruction

Usage:
    accelerate launch -m lmms_eval \
        --model star_7b_visual_cot \
        --model_args pretrained=/path/to/STAR-7B \
        --tasks mathvista_visual_cot \
        --batch_size 1 \
        --output_path ./logs/

Uni-MMMU Interleaved Mode:
    Supports special interleaved generation for Uni-MMMU tasks:
    - Jigsaw: gen_image(cand0) → gen_image(cand1) → gen_text(answer)
    - Maze/Sliding: [gen_image(step)]×k → gen_text(answer)
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from accelerate import Accelerator
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

# Add STAR repository to Python path
wd = Path(__file__).parent.parent.parent.parent.resolve()
star_path = os.path.join(str(wd), "STAR")
if os.path.exists(star_path):
    sys.path.append(star_path)
    eval_logger.info(f"Added STAR path to sys.path: {star_path}")
else:
    eval_logger.warning(
        f"STAR repository not found at {star_path}. "
        f"Please clone it: cd {wd} && git clone "
        "https://github.com/MM-MVR/STAR.git"
    )


@register_model("star_7b_visual_cot")
class STAR7BVisualCoT(lmms):
    """
    STAR-7B Visual Chain-of-Thought Model

    Performs two-stage visual reasoning:
    1. Generate visualization image from text prompt (using STAR generation mode)
    2. Answer question using the generated image (using STAR understanding mode)
    """

    def __init__(
        self,
        pretrained: str,
        # Stage 1: Image generation parameters
        stage1_vq_image_size: int = 384,
        stage1_vq_tokens: int = 576,
        stage1_topk: int = 2000,
        stage1_cfg: float = 20.0,
        stage1_topp: float = 1.0,
        # Stage 2: Visual understanding parameters
        stage2_max_new_tokens: int = 512,
        # Generation prompt template
        generation_prompt_template: str = "Generate a detailed visual diagram or illustration to help answer this question: {question}",
        # Output and debugging
        output_dir: Optional[str] = None,
        save_intermediate: bool = False,
        intermediate_dir: Optional[str] = None,
        # Error handling
        fail_gracefully: bool = True,
        # Model loading
        seed: int = 0,
        continual_mode: bool = False,
        response_persistent_folder: Optional[str] = None,
        # STAR model args
        max_pixels: int = 28 * 28 * 1024,
        min_pixels: int = 28 * 28 * 16,
        max_seq_length: int = 8192,
        max_text_tokens: int = 512,
        grad_ckpt: bool = False,
        diffusion_as_decoder: bool = False,
        diffusion_resolution: int = 1024,
        ori_inp_dit: str = "seq",
        **kwargs,
    ) -> None:
        super().__init__()

        self.pretrained = pretrained
        self.save_intermediate = save_intermediate
        self.fail_gracefully = fail_gracefully
        self.seed = seed
        self.generation_prompt_template = generation_prompt_template
        self.continual_mode = continual_mode

        # Stage 1 parameters (generation)
        self.stage1_vq_image_size = stage1_vq_image_size
        self.stage1_vq_tokens = stage1_vq_tokens
        self.stage1_topk = stage1_topk
        self.stage1_cfg = stage1_cfg
        self.stage1_topp = stage1_topp

        # Stage 2 parameters (understanding)
        self.stage2_max_new_tokens = stage2_max_new_tokens

        # Import STAR dependencies
        try:
            from star.models.config import (
                STARMultiModalConfig,
                load_config_from_json,
            )
            from star.models.model import STARMultiModal

            self.STARMultiModalConfig = STARMultiModalConfig
            self.load_config_from_json = load_config_from_json
            self.STARMultiModal = STARMultiModal

        except Exception as e:
            raise ImportError(
                f"Failed to import STAR dependencies. "
                f"Please ensure:\\n"
                f"  1. STAR repository is cloned at lmms-eval root: "
                f"git clone https://github.com/MM-MVR/STAR.git\\n"
                f"  2. Model weights are downloaded\\n"
                f"Error: {e}"
            )

        # Setup output directories
        if output_dir is None:
            self.output_dir = "./logs/star_7b_visual_cot"
        else:
            self.output_dir = output_dir

        if intermediate_dir is None:
            self.intermediate_dir = self.output_dir
        else:
            self.intermediate_dir = intermediate_dir

        self.generated_images_dir = os.path.join(self.intermediate_dir, "generated_images")
        os.makedirs(self.generated_images_dir, exist_ok=True)

        if save_intermediate:
            os.makedirs(self.intermediate_dir, exist_ok=True)
            eval_logger.info(f"Intermediate artifacts will be saved under: {self.intermediate_dir}")

        # Setup response cache for continual mode
        if response_persistent_folder is None:
            self.response_persistent_folder = os.path.join(self.output_dir, "persistent_folder")
        else:
            self.response_persistent_folder = response_persistent_folder

        self.response_cache = {}
        self.cache_mode = "start"

        if self.continual_mode:
            os.makedirs(self.response_persistent_folder, exist_ok=True)
            self.response_persistent_file = os.path.join(
                self.response_persistent_folder, "star_visual_cot_response.json"
            )

            if os.path.exists(self.response_persistent_file):
                with open(self.response_persistent_file, "r") as f:
                    self.response_cache = json.load(f)
                self.cache_mode = "resume"
                eval_logger.info(
                    f"Loaded cache: {len(self.response_cache)} records"
                )

        # Setup accelerator
        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            if self.continual_mode:
                eval_logger.warning(
                    "Continual mode is not supported for distributed inference. "
                    "Automatically disabling continual_mode."
                )
                self.continual_mode = False
            self.accelerator = accelerator
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.accelerator = accelerator
            self._rank = 0
            self._world_size = 1

        # Create args object for STAR model
        class Args:
            pass

        self.args = Args()
        self.args.max_pixels = max_pixels
        self.args.min_pixels = min_pixels
        self.args.max_seq_length = max_seq_length
        self.args.max_text_tokens = max_text_tokens
        self.args.grad_ckpt = grad_ckpt
        self.args.diffusion_as_decoder = diffusion_as_decoder
        self.args.diffusion_resolution = diffusion_resolution
        self.args.ori_inp_dit = ori_inp_dit
        # Add vq_image_size for image editing transforms
        self.args.vq_image_size = stage1_vq_image_size
        # Set to "both" mode to enable both generation and understanding
        self.args.data_type = "both"

        # Load model
        eval_logger.info(f"Loading STAR model from {pretrained}")
        self._load_model()

        # Monkey patch inference_understand to support multiple images
        self._patch_inference_understand()

        eval_logger.info("STAR Visual CoT model initialized successfully")

    def _patch_inference_understand(self):
        """Monkey patch STAR's inference_understand to support multiple images"""
        original_inference_understand = self._star_model.inference_understand

        def patched_inference_understand(image, question, max_new_tokens=256):
            """
            Enhanced inference_understand that supports multiple images

            Args:
                image: Single PIL Image, list of PIL Images, or None
                question: Text question
                max_new_tokens: Maximum tokens to generate

            Returns:
                Generated answer text
            """
            # Handle different image input types
            content = []

            if image is None:
                # Text-only input
                content.append({"type": "text", "text": question})
            elif isinstance(image, list):
                # Multiple images
                for img in image:
                    if img is not None:
                        pil_image = self._star_model.preprocess_image(img)
                        content.append({"type": "image", "image": pil_image})
                content.append({"type": "text", "text": question})
            else:
                # Single image - use original method
                return original_inference_understand(image, question, max_new_tokens)

            # Multi-image or text-only path
            messages = [{"role": "user", "content": content}]

            from qwen_vl_utils import process_vision_info

            # Preparation for inference
            text = self._star_model.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self._star_model.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self._star_model.llm.device)

            # Inference: Generation of the output
            generated_ids = self._star_model.llm.generate(
                **inputs, max_new_tokens=max_new_tokens
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self._star_model.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            return output_text[0] if output_text else ""

        # Apply the patch
        self._star_model.inference_understand = patched_inference_understand
        eval_logger.info("Patched inference_understand to support multiple images")

    def _load_model(self):
        """Load STAR model using the same pattern as inference scripts"""
        model_path = self.pretrained

        # Load configuration
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")

        eval_logger.info(f"Loading config from {config_path}")
        config_data = self.load_config_from_json(config_path)

        # Convert all relative paths in config to absolute paths
        # Language model path
        if "language_model" in config_data and "model_path" in config_data["language_model"]:
            lm_path = config_data["language_model"]["model_path"]
            if not os.path.isabs(lm_path):
                lm_path = os.path.join(model_path, lm_path)

            # Check if exists, fallback to HuggingFace
            if not os.path.exists(lm_path):
                eval_logger.warning(f"Language model not found at {lm_path}")
                config_data["language_model"]["model_path"] = "Qwen/Qwen2.5-VL-7B-Instruct"
                eval_logger.info("Using HuggingFace: Qwen/Qwen2.5-VL-7B-Instruct")
            else:
                config_data["language_model"]["model_path"] = lm_path
                eval_logger.info(f"Language model path: {lm_path}")

        # VQ model path
        if "pixel_encoder" in config_data and "model_path" in config_data["pixel_encoder"]:
            vq_path = config_data["pixel_encoder"]["model_path"]
            if not os.path.isabs(vq_path):
                vq_path = os.path.join(model_path, vq_path)

            # Check if exists, download from HuggingFace if not
            if not os.path.exists(vq_path):
                eval_logger.warning(f"VQ model not found at {vq_path}")
                eval_logger.info("Downloading VQ model from HuggingFace: MM-MVR/STAR-VQ")
                try:
                    from huggingface_hub import hf_hub_download
                    vq_path = hf_hub_download(
                        repo_id="MM-MVR/STAR-VQ",
                        filename="VQ-Model.pt",
                        cache_dir=os.path.join(model_path, ".cache")
                    )
                    config_data["pixel_encoder"]["model_path"] = vq_path
                    eval_logger.info(f"Downloaded VQ model to: {vq_path}")
                except Exception as e:
                    eval_logger.error(f"Failed to download VQ model: {e}")
                    raise
            else:
                config_data["pixel_encoder"]["model_path"] = vq_path
                eval_logger.info(f"VQ model path: {vq_path}")

        # Pixel decoder path (only needed if diffusion_as_decoder is True)
        if "pixel_decoder" in config_data and "model_path" in config_data["pixel_decoder"]:
            decoder_path = config_data["pixel_decoder"]["model_path"]
            if not os.path.isabs(decoder_path):
                decoder_path = os.path.join(model_path, decoder_path)

            if self.args.diffusion_as_decoder:
                if not os.path.exists(decoder_path):
                    eval_logger.warning(f"Pixel decoder not found at {decoder_path}")
                    config_data["pixel_decoder"]["model_path"] = "Alpha-VLLM/Lumina-Image-2.0"
                    eval_logger.info("Using HuggingFace: Alpha-VLLM/Lumina-Image-2.0")
                else:
                    config_data["pixel_decoder"]["model_path"] = decoder_path
                    eval_logger.info(f"Pixel decoder path: {decoder_path}")
            else:
                eval_logger.info("Pixel decoder not needed (diffusion_as_decoder=False)")

        model_config = self.STARMultiModalConfig(**config_data)

        # Initialize STAR model
        eval_logger.info("Initializing STAR model...")
        self._star_model = self.STARMultiModal(model_config, self.args)

        # Load checkpoint - REQUIRED for both understanding and generation modes
        checkpoint_path = os.path.join(model_path, "STAR-7B.pt")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found at {checkpoint_path}. "
                f"STAR-7B.pt is required to load trained weights. "
                f"Please download it to {model_path}"
            )

        eval_logger.info(f"Loading STAR checkpoint from {checkpoint_path}")
        with torch.no_grad():
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            # Load with strict=False to allow missing keys for skipped components
            missing_keys, unexpected_keys = self._star_model.load_state_dict(
                checkpoint["state_dict"], strict=False
            )
            if missing_keys:
                eval_logger.debug(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                eval_logger.debug(f"Unexpected keys: {unexpected_keys}")
        eval_logger.info("STAR checkpoint loaded successfully")

        # Move model to device
        device = torch.device(
            f"cuda:{self._rank}" if torch.cuda.is_available() else "cpu"
        )
        self._device = device
        self._star_model = self._star_model.to(device).to(torch.bfloat16)
        self._star_model.eval()

        eval_logger.info(f"Model loaded on device: {device}")

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    @property
    def model(self):
        return self._star_model

    @property
    def tokenizer(self):
        return self._star_model.tokenizer

    @property
    def processor(self):
        return self._star_model.processor

    @property
    def config(self):
        return self._star_model.config

    def set_seed(self, seed: int):
        """Set random seeds for reproducibility"""
        if seed > 0:
            import random

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

    def flatten(self, input_list):
        """Flatten a nested list"""
        output = []
        for item in input_list:
            if isinstance(item, list):
                output.extend(self.flatten(item))
            else:
                output.append(item)
        return output

    def _stage1_generate_image(
        self, generation_prompt: str, doc_id: str, task: str, original_image=None
    ) -> Tuple[str, List[str]]:
        """
        Stage 1: Generate visualization image from prompt

        Args:
            generation_prompt: Text prompt for image generation/editing
            doc_id: Document ID for file naming
            task: Task name for file naming
            original_image: Original image to condition on (optional)
                - If provided: Uses STAR's image editing (i2i) capability
                - If None: Uses STAR's text-to-image generation

        Returns:
            Tuple of (generated_text, list_of_image_paths)
        
        Note:
            STAR supports both text-to-image generation and image-to-image editing.
            When original_image is provided, it uses generate_images_edit for better
            context-aware visualization generation.
        """
        eval_logger.debug(f"Stage 1 - Generating image for doc {doc_id}")
        eval_logger.debug(f"Generation prompt: {generation_prompt}")

        try:
            self.set_seed(self.seed)

            # Convert original_image to PIL Image if provided
            pil_image = None
            if original_image is not None:
                pil_image = self._extract_image_from_various_formats(original_image)
                if pil_image is not None:
                    eval_logger.debug("Stage 1 - Using image editing mode (i2i) with original image")
                else:
                    eval_logger.warning("Stage 1 - Failed to extract original image, falling back to text-to-image")

            # Choose generation method based on whether we have an input image
            with torch.no_grad():
                if pil_image is not None:
                    # Image-to-image editing mode
                    # STAR's generate_images_edit expects a list of images
                    output = self._star_model.generate_images_edit(
                        image=[pil_image],  # Wrap single image in list
                        prompt=generation_prompt,
                        max_new_tokens=self.stage1_vq_tokens,
                        num_return_sequences=1,
                        cfg_weight=self.stage1_cfg,
                        topk_sample=self.stage1_topk,
                        topp_sample=self.stage1_topp,
                        return_dict=True,
                    )
                else:
                    # Text-to-image generation mode
                    output = self._star_model.generate_images(
                        generation_prompt,
                        max_new_tokens=self.stage1_vq_tokens,
                        num_return_sequences=1,
                        cfg_weight=self.stage1_cfg,
                        topk_sample=self.stage1_topk,
                        topp_sample=self.stage1_topp,
                        return_dict=True,
                    )

            # Process and save images
            image_paths = []
            if output is not None and isinstance(output, dict):
                output_images = output.get("output_images")
                diff_images = output.get("diff_images")

                # Create task-specific directory
                task_dir = os.path.join(self.generated_images_dir, task)
                os.makedirs(task_dir, exist_ok=True)

                # Save VQ images
                if output_images is not None:
                    dec_vq = np.clip((output_images + 1) / 2 * 255, 0, 255)
                    visual_img_vq = np.zeros(
                        (1, self.stage1_vq_image_size, self.stage1_vq_image_size, 3),
                        dtype=np.uint8,
                    )
                    visual_img_vq[0, :, :] = dec_vq[0]
                    img = Image.fromarray(visual_img_vq[0].astype(np.uint8))

                    img_filename = f"{doc_id}_stage1_vq.png"
                    img_path = os.path.join(task_dir, img_filename)
                    img.save(img_path)
                    image_paths.append(img_path)
                    eval_logger.debug(f"Stage 1 - Saved VQ image: {img_path}")

                # Save diffusion images if available
                if diff_images is not None and len(diff_images) > 0:
                    img_filename = f"{doc_id}_stage1_diff.png"
                    img_path = os.path.join(task_dir, img_filename)
                    diff_images[0].save(img_path)
                    image_paths.append(img_path)
                    eval_logger.debug(f"Stage 1 - Saved diffusion image: {img_path}")

            output_text = f"Generated {len(image_paths)} images"
            eval_logger.debug(f"Stage 1 - Generated {len(image_paths)} image(s)")
            return output_text, image_paths

        except Exception as e:
            eval_logger.error(f"Stage 1 failed for doc {doc_id}: {e}")
            if self.fail_gracefully:
                return "", []
            else:
                raise

    def _stage2_answer_with_image(
        self, question: str, image_path: str, doc_id: str, original_image: Optional[Image.Image] = None
    ) -> str:
        """
        Stage 2: Answer question using generated image (and optionally original image)

        Args:
            question: Original question text
            image_path: Path to generated auxiliary image
            doc_id: Document ID for logging
            original_image: Original image (optional, used as primary reference)

        Returns:
            Answer text
        
        Note:
            This method supports multi-image input through the patched inference_understand.
            When original_image is provided, both original and generated images are used.
            Images are processed in order: [original_image, auxiliary_image]
        """
        eval_logger.debug(f"Stage 2 - Answering question for doc {doc_id}")
        eval_logger.debug(f"Question: {question}")

        try:
            self.set_seed(self.seed)

            # Load generated auxiliary image
            auxiliary_image = Image.open(image_path).convert("RGB")

            # Prepare images list for multi-image input
            images = []
            if original_image is not None:
                eval_logger.debug("Stage 2 - Using both original and auxiliary images")
                # Convert original_image if needed
                if not isinstance(original_image, Image.Image):
                    original_image = self._extract_image_from_various_formats(original_image)
                if original_image is not None:
                    images.append(original_image)
            
            images.append(auxiliary_image)
            
            eval_logger.debug(f"Stage 2 - Processing with {len(images)} image(s)")

            # Use patched inference_understand with multiple images
            # If only one image, pass it directly; if multiple, pass as list
            with torch.no_grad():
                answer = self._star_model.inference_understand(
                    image=images if len(images) > 1 else images[0],
                    question=question,
                    max_new_tokens=self.stage2_max_new_tokens
                )

            eval_logger.debug(f"Stage 2 - Generated answer: {answer[:100]}...")
            return answer

        except Exception as e:
            eval_logger.error(f"Stage 2 failed for doc {doc_id}: {e}")
            if self.fail_gracefully:
                return ""
            else:
                raise

    def _extract_image_from_various_formats(self, img_data) -> Optional[Image.Image]:
        """
        Extract PIL Image from various formats (HuggingFace datasets, file paths, etc.)
        
        Args:
            img_data: Image data in various formats
            
        Returns:
            PIL Image or None if extraction fails
        """
        try:
            if img_data is None:
                return None
            elif isinstance(img_data, Image.Image):
                return img_data.convert("RGB")
            elif isinstance(img_data, str):
                # File path
                return Image.open(img_data).convert("RGB")
            elif isinstance(img_data, dict):
                # HuggingFace dataset format
                if "bytes" in img_data:
                    from io import BytesIO
                    return Image.open(BytesIO(img_data["bytes"])).convert("RGB")
                elif "path" in img_data:
                    return Image.open(img_data["path"]).convert("RGB")
                elif "image" in img_data:
                    inner_img = img_data["image"]
                    return self._extract_image_from_various_formats(inner_img)
            else:
                # Try to open it directly
                return Image.open(img_data).convert("RGB")
        except Exception as e:
            eval_logger.debug(f"Failed to extract image from format {type(img_data)}: {e}")
            return None

    def _save_intermediate_artifacts(
        self,
        doc_id: str,
        task: str,
        generation_prompt: str,
        stage1_text: str,
        generated_images: List[str],
        question: str,
        stage2_answer: str,
    ) -> None:
        """Save intermediate artifacts for debugging"""
        if not self.save_intermediate:
            return

        artifact_dir = os.path.join(self.intermediate_dir, task)
        os.makedirs(artifact_dir, exist_ok=True)

        # Save metadata
        metadata = {
            "doc_id": doc_id,
            "task": task,
            "generation_prompt": generation_prompt,
            "stage1_text": stage1_text,
            "generated_images": generated_images,
            "question": question,
            "stage2_answer": stage2_answer,
        }

        metadata_path = os.path.join(artifact_dir, f"{doc_id}_metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        eval_logger.debug(f"Saved intermediate artifacts to: {metadata_path}")

    def generate_uni_mmmu_interleaved(
        self,
        input_images: List,
        prompt: str,
        doc_id: str,
        task: str,
        interleaved_config: dict,
        doc: dict = None,
    ) -> Tuple[str, List[str]]:
        """
        Uni-MMMU interleaved generation for STAR Visual CoT.

        This implements the exact generation flow from the original Uni-MMMU:
        - Jigsaw: gen_image(cand0) → gen_image(cand1) → gen_text(answer)
        - Maze/Sliding: [gen_text(plan) → gen_image(step)]×k → gen_text(answer)

        Args:
            input_images: List of input images
            prompt: Base prompt text
            doc_id: Document ID for file naming
            task: Task name for file naming
            interleaved_config: Configuration dict from yaml
            doc: Document data for dynamic num_images extraction

        Returns:
            Tuple of (final_text_answer, list_of_generated_image_paths)
        """
        task_type = interleaved_config.get("task_type", "jigsaw")

        # Get num_images dynamically from doc if available
        num_images = interleaved_config.get("num_images", 2)
        if doc is not None:
            if task_type == "maze":
                # Get step count from ground truth
                steps_str = doc.get("steps", "[]")
                steps = json.loads(steps_str) if isinstance(steps_str, str) else steps_str
                if steps:
                    num_images = len(steps)
            elif task_type == "sliding":
                # Get step count from ground truth
                steps_str = doc.get("steps_words", "[]")
                steps = json.loads(steps_str) if isinstance(steps_str, str) else steps_str
                if steps:
                    num_images = len(steps)

        # Extract original image from input_images
        original_image = None
        if input_images and len(input_images) > 0:
            original_image = self._extract_image_from_various_formats(input_images[0])

        generated_images = []

        if task_type == "jigsaw":
            # Jigsaw: Generate 2 completed images then final answer
            # Image 1: Candidate 0 completion
            suffix1 = "Output ONLY a single image with Candidate 0 placed in the bottom-right cell. No text."
            gen_prompt1 = prompt + "\n\n" + suffix1

            _, img_paths_0 = self._stage1_generate_image(
                generation_prompt=gen_prompt1,
                doc_id=f"{doc_id}_cand0",
                task=task,
                original_image=original_image,
            )
            if img_paths_0:
                generated_images.extend(img_paths_0)
                eval_logger.info(f"Saved jigsaw image 0: {img_paths_0[0]}")

            # Image 2: Candidate 1 completion
            suffix2 = "Output ONLY a single image with Candidate 1 placed in the bottom-right cell. No text."
            gen_prompt2 = prompt + "\n\n" + suffix2

            _, img_paths_1 = self._stage1_generate_image(
                generation_prompt=gen_prompt2,
                doc_id=f"{doc_id}_cand1",
                task=task,
                original_image=original_image,
            )
            if img_paths_1:
                generated_images.extend(img_paths_1)
                eval_logger.info(f"Saved jigsaw image 1: {img_paths_1[0]}")

            # Final answer using stage 2 with all generated images
            final_suffix = (
                'Now output EXACTLY ONE <FINAL_ANSWER_JSON>{"choice": 0 or 1, "rationale": "≤30 words"}</FINAL_ANSWER_JSON>\n'
                "Do not output any additional images."
            )
            final_question = prompt + "\n\n" + final_suffix

            # Use stage 2 to answer with the generated images
            if len(generated_images) >= 2:
                # Load both generated images
                gen_img0 = Image.open(generated_images[0]).convert("RGB")
                gen_img1 = Image.open(generated_images[1]).convert("RGB")

                # Build images list
                images = []
                if original_image:
                    images.append(original_image)
                images.extend([gen_img0, gen_img1])

                # Use stage 2 with multiple images
                final_text = self._stage2_answer_with_image(
                    question=final_question,
                    image_path=generated_images[0],  # Dummy, we'll use images list
                    doc_id=doc_id,
                    original_image=images[0] if len(images) > 0 else None
                )
            else:
                final_text = ""

        else:
            # Maze/Sliding: [gen_text(plan) → gen_image(step)]×k → gen_text(answer)
            for i in range(1, num_images + 1):
                # Generate step image with planning prompt
                if task_type == "maze":
                    plan_suffix = f'Step {i}: Generate an image showing the next move (one step up/down/left/right).'
                else:  # sliding
                    plan_suffix = f'Step {i}: Generate an image showing which tile to move and in which direction.'

                gen_prompt = prompt + "\n\n" + plan_suffix

                _, img_paths = self._stage1_generate_image(
                    generation_prompt=gen_prompt,
                    doc_id=f"{doc_id}_step_{i:04d}",
                    task=task,
                    original_image=original_image,
                )

                if img_paths:
                    generated_images.extend(img_paths)
                    eval_logger.info(f"Saved step {i} image: {img_paths[0]}")

            # Final answer using all generated step images
            final_suffix = (
                "After the images, emit EXACTLY ONE LINE containing ONLY the final move list "
                "as <ANSWER_JSON>[...]</ANSWER_JSON>. No other text."
            )
            final_question = prompt + "\n\n" + final_suffix

            # Use stage 2 to answer with all generated images
            if generated_images:
                # Load all generated images
                step_images = [Image.open(img_path).convert("RGB") for img_path in generated_images]

                # Build images list
                images = []
                if original_image:
                    images.append(original_image)
                images.extend(step_images)

                # Use patched inference_understand with multiple images
                self.set_seed(self.seed)
                with torch.no_grad():
                    final_text = self._star_model.inference_understand(
                        image=images,
                        question=final_question,
                        max_new_tokens=self.stage2_max_new_tokens
                    )
            else:
                final_text = ""

        return final_text, generated_images

    def generate_until(self, requests: List[Instance]) -> List[str]:
        """
        Main inference method implementing two-stage visual CoT

        Stage 1: Generate visualization image from text prompt
        Stage 2: Answer question using the generated image
        """
        res = []
        pbar = tqdm(
            total=len(requests),
            disable=(self.rank != 0),
            desc="STAR Visual CoT Generating",
        )

        def get_uuid(task, split, doc_id):
            return f"{task}___{split}___{doc_id}"

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [
            reg.args for reg in requests
        ]:
            doc_uuid = get_uuid(task, split, doc_id)

            # Check cache
            if self.continual_mode and self.cache_mode == "resume":
                if doc_uuid in self.response_cache:
                    content = self.response_cache[doc_uuid]
                    if content:
                        res.append(content)
                        pbar.update(1)
                        continue

            # Check if this is Uni-MMMU interleaved generation mode
            bagel_interleaved = gen_kwargs.get("bagel_interleaved", None)

            if bagel_interleaved is not None:
                # Uni-MMMU interleaved generation mode
                eval_logger.info(f"Uni-MMMU interleaved mode for doc {doc_id}")

                # Get input images and doc data
                doc = self.task_dict[task][split][doc_id]
                input_images = []
                if doc_to_visual:
                    visuals = doc_to_visual(doc)
                    if visuals:
                        input_images = visuals if isinstance(visuals, list) else [visuals]

                # Generate using interleaved mode
                final_answer, gen_paths = self.generate_uni_mmmu_interleaved(
                    input_images, contexts, str(doc_id), task, bagel_interleaved, doc
                )

                # Save intermediate artifacts if enabled
                self._save_intermediate_artifacts(
                    doc_id=str(doc_id),
                    task=task,
                    generation_prompt=f"Interleaved generation: {bagel_interleaved.get('task_type', 'unknown')}",
                    stage1_text="",
                    generated_images=gen_paths,
                    question=contexts,
                    stage2_answer=final_answer,
                )

                res.append(final_answer)

                # Update cache
                if self.continual_mode:
                    self.response_cache[doc_uuid] = final_answer
                    with open(self.response_persistent_file, "w") as f:
                        json.dump(
                            self.response_cache, f, ensure_ascii=False, indent=2
                        )

                pbar.update(1)
                continue

            # Standard single-image generation mode
            # Extract original image from document
            original_image = None
            if doc_to_visual is not None:
                try:
                    visuals = doc_to_visual(self.task_dict[task][split][doc_id])
                    if isinstance(visuals, list):
                        visuals = self.flatten(visuals)
                        if visuals and len(visuals) > 0:
                            original_image = visuals[0]
                    elif visuals is not None:
                        original_image = visuals
                    if original_image is not None:
                        eval_logger.debug(f"Extracted original image for doc {doc_id}")
                except Exception as e:
                    eval_logger.warning(f"Failed to extract original image for doc {doc_id}: {e}")

            # Parse contexts to extract generation_prompt if provided
            import re
            gen_prompt_match = re.search(r'\[GEN_PROMPT\](.*?)\[/GEN_PROMPT\]', contexts, re.DOTALL)
            question_match = re.search(r'\[QUESTION\](.*?)\[/QUESTION\]', contexts, re.DOTALL)

            if gen_prompt_match and question_match:
                # Use custom generation prompt from task config
                custom_gen_prompt = gen_prompt_match.group(1).strip()
                actual_question = question_match.group(1).strip()
                generation_prompt = custom_gen_prompt.replace("{question}", actual_question)
                # Update contexts to be just the question for stage 2
                contexts = contexts.replace(f"[GEN_PROMPT]{gen_prompt_match.group(1)}[/GEN_PROMPT]", "")
                contexts = contexts.replace(f"[QUESTION]{question_match.group(1)}[/QUESTION]", question_match.group(1))
                eval_logger.info("Using custom generation prompt from task config")
            else:
                # Use default template
                generation_prompt = self.generation_prompt_template.format(question=contexts)

            eval_logger.info(f"\n{'='*60}")
            eval_logger.info(f"Processing doc {doc_id} from task {task}")
            eval_logger.info(f"{'='*60}")

            # Stage 1: Generate visualization image
            stage1_text, generated_images = self._stage1_generate_image(
                generation_prompt=generation_prompt,
                doc_id=doc_id,
                task=task,
                original_image=original_image,
            )

            # Check if image was generated
            if not generated_images or len(generated_images) == 0:
                eval_logger.warning(
                    f"No image generated for doc {doc_id}, using stage 1 text as answer"
                )
                res.append(stage1_text if stage1_text else "")
                pbar.update(1)
                continue

            # Stage 2: Answer question using generated image (and original image if available)
            final_answer = self._stage2_answer_with_image(
                question=contexts,
                image_path=generated_images[0],
                doc_id=doc_id,
                original_image=original_image
            )

            # Save intermediate artifacts if enabled
            self._save_intermediate_artifacts(
                doc_id=doc_id,
                task=task,
                generation_prompt=generation_prompt,
                stage1_text=stage1_text,
                generated_images=generated_images,
                question=contexts,
                stage2_answer=final_answer,
            )

            # Return only final answer text
            res.append(final_answer)

            # Update cache
            if self.continual_mode:
                self.response_cache[doc_uuid] = final_answer
                with open(self.response_persistent_file, "w") as f:
                    json.dump(
                        self.response_cache, f, ensure_ascii=False, indent=2
                    )

            pbar.update(1)

        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """Not supported for generation models"""
        raise NotImplementedError(
            "STAR Visual CoT is a generation model and does not support loglikelihood"
        )

    def generate_until_multi_round(self, requests) -> List[str]:
        """Multi-round dialogue generation"""
        raise NotImplementedError(
            "Multi-round dialogue not yet implemented for STAR Visual CoT"
        )