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


@register_model("star_7b")
class STAR7B(lmms):
    """
    STAR-7B: STacked AutoRegressive Scheme for Unified Multimodal Learning

    Supports both image understanding and text-to-image generation using
    a stacked autoregressive architecture with VQ-VAE encoding.

    Modes:
        - "understanding": Visual understanding (image + text -> text)
        - "generation": Image generation (text -> image)

    Example usage for understanding:
    accelerate launch -m lmms_eval \\
        --model star_7b \\
        --model_args pretrained=/path/to/STAR-7B,mode=understanding \\
        --tasks mmbench \\
        --batch_size 1 \\
        --output_path ./logs/

    Example usage for generation:
    accelerate launch -m lmms_eval \\
        --model star_7b \\
        --model_args pretrained=/path/to/STAR-7B,mode=generation \\
        --tasks ueval \\
        --batch_size 1 \\
        --output_path ./logs/
    """

    def __init__(
        self,
        pretrained: str,
        mode: str = "understanding",
        output_image_dir: Optional[str] = None,
        max_new_tokens: int = 512,
        seed: int = 0,
        continual_mode: bool = True,
        response_persistent_folder: Optional[str] = None,
        # STAR model args
        max_pixels: int = 28 * 28 * 1024,
        min_pixels: int = 28 * 28 * 16,
        max_seq_length: int = 8192,
        max_text_tokens: int = 512,
        grad_ckpt: bool = False,
        diffusion_as_decoder: bool = False,
        # Generation args
        vq_image_size: int = 384,
        vq_tokens: int = 576,
        topk: int = 2000,
        cfg: float = 20.0,
        topp: float = 1.0,
        diffusion_resolution: int = 1024,
        ori_inp_dit: str = "seq",
        **kwargs,
    ) -> None:
        super().__init__()

        # Validate mode
        if mode not in ["understanding", "generation"]:
            raise ValueError(
                f"mode must be 'understanding' or 'generation', got '{mode}'"
            )

        self.mode = mode
        self.max_new_tokens = max_new_tokens
        self.pretrained = pretrained
        self.seed = seed
        self.continual_mode = continual_mode

        # Generation parameters
        self.vq_image_size = vq_image_size
        self.vq_tokens = vq_tokens
        self.topk = topk
        self.cfg = cfg
        self.topp = topp

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

        # Setup output directory
        if response_persistent_folder is None:
            self.response_persistent_folder = "./logs/star_persistent_folder"
        else:
            self.response_persistent_folder = response_persistent_folder

        if output_image_dir is None:
            self.output_image_dir = os.path.join(
                self.response_persistent_folder, "star_generated_images"
            )
        else:
            self.output_image_dir = output_image_dir

        os.makedirs(self.output_image_dir, exist_ok=True)
        eval_logger.info(f"Image output directory: {self.output_image_dir}")

        # Setup response cache for continual mode
        self.response_cache = {}
        self.cache_mode = "start"

        if self.continual_mode:
            os.makedirs(self.response_persistent_folder, exist_ok=True)
            self.response_persistent_file = os.path.join(
                self.response_persistent_folder, "star_response.json"
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

        # Create args object for STAR model (mimicking inference scripts)
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
        self.args.data_type = mode

        # Load model using STAR's model_setup pattern
        eval_logger.info(f"Loading STAR model from {pretrained}")
        self._load_model()

        # Monkey patch inference_understand to support multiple images
        self._patch_inference_understand()

        eval_logger.info("STAR model initialized successfully")

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
        # The checkpoint contains trained weights from STAR's 4-stage training
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
                eval_logger.debug(f"Missing keys (expected in understanding mode): {missing_keys}")
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

    def understand_image(self, prompt: str, images, doc_id: str) -> str:
        """
        Understand image(s) and answer question using STAR's inference_understand API

        Args:
            prompt: Input text prompt/question
            images: PIL Image, list of PIL Images, or None
            doc_id: Document ID for logging

        Returns:
            Generated text answer
        """
        self.set_seed(self.seed)

        # Handle different image input types
        if images is None:
            # Text-only input
            eval_logger.warning(f"No images provided for doc_id={doc_id}, using text-only")
            image = None
        elif isinstance(images, list):
            if len(images) == 0:
                eval_logger.warning(f"Empty image list for doc_id={doc_id}")
                image = None
            else:
                # Pass the entire list to the patched inference_understand
                image = images
        else:
            # Single image
            image = images

        # Use STAR's native inference_understand method (patched to support multi-images)
        with torch.no_grad():
            answer = self._star_model.inference_understand(
                image=image, question=prompt, max_new_tokens=self.max_new_tokens
            )

        return answer

    def generate_images_from_prompt(
        self, prompt: str, doc_id: str, task: str
    ) -> Tuple[str, List[str]]:
        """
        Generate images from text prompt using STAR's generate_images API

        Args:
            prompt: Input text prompt
            doc_id: Document ID for file naming
            task: Task name for file naming

        Returns:
            Tuple of (generated_text, list_of_image_paths)
        """
        self.set_seed(self.seed)

        # Use STAR's native generate_images method
        with torch.no_grad():
            output = self._star_model.generate_images(
                prompt,
                max_new_tokens=self.vq_tokens,
                num_return_sequences=1,
                cfg_weight=self.cfg,
                topk_sample=self.topk,
                topp_sample=self.topp,
                return_dict=True,
            )

        # Process and save images
        image_paths = []
        if output is not None and isinstance(output, dict):
            output_images = output.get("output_images")
            diff_images = output.get("diff_images")

            # Save VQ images
            if output_images is not None:
                dec_vq = np.clip((output_images + 1) / 2 * 255, 0, 255)
                visual_img_vq = np.zeros(
                    (1, self.vq_image_size, self.vq_image_size, 3),
                    dtype=np.uint8,
                )
                visual_img_vq[0, :, :] = dec_vq[0]
                img = Image.fromarray(visual_img_vq[0].astype(np.uint8))

                img_filename = f"{task}_{doc_id}_vq.png"
                img_path = os.path.join(self.output_image_dir, img_filename)
                img.save(img_path)
                image_paths.append(img_path)

            # Save diffusion images if available
            if diff_images is not None and len(diff_images) > 0:
                img_filename = f"{task}_{doc_id}_diff.png"
                img_path = os.path.join(self.output_image_dir, img_filename)
                diff_images[0].save(img_path)
                image_paths.append(img_path)

        output_text = f"Generated {len(image_paths)} images"
        return output_text, image_paths

    def flatten(self, input_list):
        """Flatten a nested list"""
        output = []
        for item in input_list:
            if isinstance(item, list):
                output.extend(self.flatten(item))
            else:
                output.append(item)
        return output

    def generate_until(self, requests: List[Instance]) -> List[str]:
        """Main inference method"""
        res = []
        pbar = tqdm(
            total=len(requests),
            disable=(self.rank != 0),
            desc="STAR Generating",
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

            prompt = contexts

            if self.mode == "understanding":
                # Image understanding mode
                visuals = None

                if doc_to_visual is not None:
                    # Get images from doc_to_visual
                    try:
                        visuals = doc_to_visual(self.task_dict[task][split][doc_id])
                        # Flatten nested lists
                        if isinstance(visuals, list):
                            visuals = self.flatten(visuals)
                    except Exception as e:
                        eval_logger.warning(
                            f"Failed to load visuals for doc_id={doc_id}: {e}"
                        )
                        visuals = None

                # Call understand_image with images (can be None, single image, or list)
                output_text = self.understand_image(prompt, visuals, str(doc_id))
                formatted_output = output_text

            else:
                # Image generation mode
                output_text, output_images = self.generate_images_from_prompt(
                    prompt, str(doc_id), task
                )
                formatted_output = json.dumps(
                    {"text": output_text, "images": output_images},
                    ensure_ascii=False,
                )

            res.append(formatted_output)

            # Update cache
            if self.continual_mode:
                self.response_cache[doc_uuid] = formatted_output
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
            "STAR is a generation model and does not support loglikelihood"
        )

    def generate_until_multi_round(self, requests) -> List[str]:
        """Multi-round dialogue generation"""
        raise NotImplementedError(
            "TODO: Implement multi-round dialogue generation"
        )
