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
from transformers import AutoTokenizer

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

# Add OneCAT path to sys.path
wd = Path(__file__).parent.parent.parent.parent.resolve()
onecat_path = os.path.join(str(wd), "OneCAT")

# Try multiple possible locations for OneCAT repository
possible_paths = [
    onecat_path,  # /home/xinjiezhang/data/lei/lmms-eval/OneCAT
    os.path.join(str(wd.parent), "OneCAT"),  # /home/xinjiezhang/data/lei/OneCAT
]

onecat_found = False
for path in possible_paths:
    if os.path.exists(path):
        sys.path.append(path)
        eval_logger.info(f"Added OneCAT path to sys.path: {path}")
        onecat_found = True
        break

if not onecat_found:
    eval_logger.warning(
        f"OneCAT repository not found. Tried: {possible_paths}. "
        f"Please ensure it's in the correct location."
    )


@register_model("onecat")
class OneCAT(lmms):
    """
    OneCAT: Decoder-Only Auto-Regressive Model for Unified Understanding and Generation

    Supports visual understanding, text-to-image generation, and image editing.
    This integration focuses on visual understanding for evaluation tasks.

    Example usage:
    python -m lmms_eval \
        --model onecat \
        --model_args pretrained=/path/to/OneCAT-3B \
        --tasks illusionbench_arshia_icon_shape_test \
        --batch_size 1 \
        --output_path ./logs/
    """

    def __init__(
        self,
        pretrained: str,
        vae_path: Optional[str] = None,
        max_new_tokens: int = 1000,
        do_sample: bool = False,
        num_beams: int = 1,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        dtype: str = "bfloat16",
        device: str = "cuda",
        continual_mode: bool = True,
        response_persistent_folder: Optional[str] = None,
        simple_output: bool = True,
        simple_output_dir: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        self.pretrained = pretrained
        self.vae_path = vae_path
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.num_beams = num_beams
        self.top_k = top_k
        self.top_p = top_p
        self.continual_mode = continual_mode
        self.device_str = device

        # Determine dtype
        if dtype == "bfloat16":
            self.torch_dtype = torch.bfloat16
        elif dtype == "float16":
            self.torch_dtype = torch.float16
        else:
            self.torch_dtype = torch.float32

        # Setup response cache for continual mode
        self.response_cache = {}
        self.cache_mode = "start"

        if response_persistent_folder is None:
            self.response_persistent_folder = "./logs/onecat_persistent_folder"
        else:
            self.response_persistent_folder = response_persistent_folder

        if self.continual_mode:
            os.makedirs(self.response_persistent_folder, exist_ok=True)
            self.response_persistent_file = os.path.join(
                self.response_persistent_folder, "onecat_response.json"
            )

            if os.path.exists(self.response_persistent_file):
                with open(self.response_persistent_file) as f:
                    self.response_cache = json.load(f)
                self.cache_mode = "resume"
                eval_logger.info(
                    f"Loaded cache: {len(self.response_cache)} records"
                )

        # Setup simple output
        self.simple_output = simple_output
        if simple_output_dir is None:
            self.simple_output_dir = "./logs/onecat_simple_output"
        else:
            self.simple_output_dir = simple_output_dir

        if self.simple_output:
            os.makedirs(self.simple_output_dir, exist_ok=True)
            os.makedirs(os.path.join(self.simple_output_dir, "images"), exist_ok=True)
            self.simple_results = []

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

        # Load model
        eval_logger.info(f"Loading OneCAT model from {pretrained}")
        self._load_model()

        eval_logger.info("OneCAT model initialized successfully")

    def _load_model(self):
        """Load OneCAT model and tokenizer"""
        try:
            from onecat.modeling_onecat import OneCatVLModel
            from onecat.smart_resize import smart_resize
            from onecat.util import build_transform

            self.OneCatVLModel = OneCatVLModel
            self.smart_resize = smart_resize
            self.build_transform = build_transform

        except Exception as e:
            raise ImportError(
                f"Failed to import OneCAT dependencies. "
                f"Please ensure:\n"
                f"  1. OneCAT repository is available at {onecat_path}\n"
                f"  2. Required dependencies are installed\n"
                f"Error: {e}"
            )

        # Load model
        self._model = self.OneCatVLModel.from_pretrained(self.pretrained)
        self._model = self._model.to(
            device=self.device_str, dtype=self.torch_dtype
        ).eval()

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self.pretrained)

        eval_logger.info(
            f"Model loaded with dtype={self.torch_dtype}, device={self.device_str}"
        )

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    @property
    def model(self):
        return self._model

    @property
    def tokenizer(self):
        return self._tokenizer

    def _load_image(self, image: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load and preprocess image for OneCAT model

        Args:
            image: PIL Image

        Returns:
            Tuple of (pixel_values, pixel_values_thumbnail)
        """
        width, height = image.size

        # Smart resize
        resized_height, resized_width = self.smart_resize(height, width)
        transform = self.build_transform(input_size=(resized_height, resized_width))
        pixel_values = transform(image).unsqueeze(0)

        # Thumbnail (base size 448x448)
        transform_base = self.build_transform(input_size=(448, 448))
        pixel_values_thumbnail = transform_base(image).unsqueeze(0)

        return pixel_values, pixel_values_thumbnail

    def understand_image(self, prompt: str, image: Image.Image) -> str:
        """
        Understand image and answer question

        Args:
            prompt: Input text prompt/question
            image: PIL Image to understand

        Returns:
            Generated text answer
        """
        # Prepare image
        pixel_values, pixel_values_thumbnail = self._load_image(image)
        pixel_values = pixel_values.to(
            device=self.device_str, dtype=self.torch_dtype
        )
        pixel_values_thumbnail = pixel_values_thumbnail.to(
            device=self.device_str, dtype=self.torch_dtype
        )

        # Generation config
        generation_config = dict(
            do_sample=self.do_sample,
            top_k=self.top_k,
            top_p=self.top_p,
            num_beams=self.num_beams,
            max_new_tokens=self.max_new_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # Generate
        response = self.model.chat(
            tokenizer=self.tokenizer,
            pixel_values=pixel_values,
            question=prompt,
            generation_config=generation_config,
            pixel_values_thumbnail=pixel_values_thumbnail,
            verbose=False,
        )

        return response

    def flatten(self, input_list):
        """Flatten a nested list"""
        output = []
        for item in input_list:
            if isinstance(item, list):
                output.extend(self.flatten(item))
            else:
                output.append(item)
        return output

    def format_output(self, text: str, images: List[str]) -> str:
        """Format output as JSON string"""
        eval_logger.debug(f"[FORMAT OUTPUT] Input: text type={type(text).__name__}, text value={repr(text)}, images count={len(images) if images else 0}")
        output_dict = {"text": text, "images": images}
        result = json.dumps(output_dict, ensure_ascii=False)
        eval_logger.debug(f"[FORMAT OUTPUT] Output JSON: {result[:200]}..." if len(result) > 200 else f"[FORMAT OUTPUT] Output JSON: {result}")
        return result

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
        Uni-MMMU interleaved generation for OneCAT.

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
        import json as json_module
        import numpy as np

        task_type = interleaved_config.get("task_type", "jigsaw")

        # Get num_images dynamically from doc if available
        num_images = interleaved_config.get("num_images", 2)
        if doc is not None:
            if task_type == "maze":
                # Get step count from ground truth
                steps_str = doc.get("steps", "[]")
                steps = json_module.loads(steps_str) if isinstance(steps_str, str) else steps_str
                if steps:
                    num_images = len(steps)
            elif task_type == "sliding":
                # Get step count from ground truth
                steps_str = doc.get("steps_words", "[]")
                steps = json_module.loads(steps_str) if isinstance(steps_str, str) else steps_str
                if steps:
                    num_images = len(steps)

        # Override generation params if specified
        cfg_scale = interleaved_config.get("cfg_scale", 20.0)
        top_k = interleaved_config.get("top_k", 2)
        top_p = interleaved_config.get("top_p", 0.97)
        h_div_w = interleaved_config.get("h_div_w", 1.0)
        t2i_stage = interleaved_config.get("t2i_stage", 3)

        # Text generation params
        text_max_new_tokens = interleaved_config.get("text_max_new_tokens", 512)
        text_do_sample = interleaved_config.get("text_do_sample", False)

        generated_images = []
        conversation_history = []

        # Load VAE for image generation
        eval_logger.info("Loading VAE for Uni-MMMU interleaved generation")
        if not hasattr(self._model, 'vae_local') or self._model.vae_local is None:
            # Import VAE loading utilities
            try:
                from onecat.var_model.tools.run_infinity import load_visual_tokenizer
                import argparse

                if t2i_stage == 2:
                    pn = "0.25M"
                elif t2i_stage == 3:
                    pn = "1M"
                else:
                    raise ValueError(f"Expected t2i_stage 2 or 3. Got {t2i_stage}")

                vae_args = argparse.Namespace(
                    vae_type=32,
                    vae_path=interleaved_config.get("vae_path", self.vae_path or ""),
                    apply_spatial_patchify=0,
                    pn=pn,
                )

                if not vae_args.vae_path:
                    raise ValueError("vae_path must be provided either in model_args or interleaved_config")

                vae_local = load_visual_tokenizer(vae_args)
                vae_local.eval()
                vae_local = vae_local.to(device=self.device_str, dtype=self.torch_dtype)
                self._model.vae_local = vae_local
                self._model.vargpt_gen_args = vae_args
                eval_logger.info(f"VAE loaded to {self.device_str}")
            except Exception as e:
                eval_logger.error(f"Failed to load VAE: {e}")
                raise

        # Setup token IDs for image generation
        try:
            from onecat.constants import IMG_GEN_CONTEXT_TOKEN, IMG_GEN_START_TOKEN, SYSTEM_PROMPT
            img_gen_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_GEN_CONTEXT_TOKEN)
            img_gen_start_token_id = self.tokenizer.convert_tokens_to_ids(IMG_GEN_START_TOKEN)
            self._model.img_gen_context_token_id = img_gen_context_token_id
            self._model.img_gen_start_token_id = img_gen_start_token_id
        except Exception as e:
            eval_logger.error(f"Failed to setup image generation tokens: {e}")
            raise

        # Add input images to conversation history
        for idx, img in enumerate(input_images):
            if img is not None:
                conversation_history.append({"type": "image", "content": img})

        # Add initial prompt
        conversation_history.append({"type": "text", "content": prompt})

        if task_type == "jigsaw":
            # Jigsaw: Generate 2 completed images then final answer
            eval_logger.info("Uni-MMMU Jigsaw mode: generating 2 candidate images")

            # Generate Candidate 0 image with full conversation context
            suffix1 = "Output ONLY a single image with Candidate 0 placed in the bottom-right cell. No text."

            img0 = self._generate_image_from_conversation(
                conversation_history, suffix1, cfg_scale, top_k, top_p, h_div_w, t2i_stage
            )
            img0_path = os.path.join(self.simple_output_dir, "images", f"{task}_{doc_id}_cand0.png")
            os.makedirs(os.path.dirname(img0_path), exist_ok=True)
            img0.save(img0_path)
            generated_images.append(img0_path)
            eval_logger.info(f"Generated Candidate 0: {img0_path}")

            # Add Candidate 0 to conversation history for generating Candidate 1
            conversation_history.append({"type": "image", "content": img0})
            conversation_history.append({"type": "text", "content": "COMPLETED WITH CANDIDATE 0:"})

            # Generate Candidate 1 image with updated conversation context
            suffix2 = "Output ONLY a single image with Candidate 1 placed in the bottom-right cell. No text."

            img1 = self._generate_image_from_conversation(
                conversation_history, suffix2, cfg_scale, top_k, top_p, h_div_w, t2i_stage
            )
            img1_path = os.path.join(self.simple_output_dir, "images", f"{task}_{doc_id}_cand1.png")
            img1.save(img1_path)
            generated_images.append(img1_path)
            eval_logger.info(f"Generated Candidate 1: {img1_path}")

            # Rebuild context for final answer generation (align with Uni-MMMU/Bagel)
            eval_logger.info("Rebuilding conversation context for final answer generation")
            conversation_history = []

            # Re-add original input images (reference 2x2 + candidate 0 patch + candidate 1 patch)
            for idx, img in enumerate(input_images):
                if img is not None:
                    conversation_history.append({"type": "image", "content": img})

            # Re-add initial prompt
            conversation_history.append({"type": "text", "content": prompt})

            # Add generated Candidate 0 image
            conversation_history.append({"type": "image", "content": img0})
            conversation_history.append({"type": "text", "content": "COMPLETED WITH CANDIDATE 0:"})

            # Add generated Candidate 1 image
            conversation_history.append({"type": "image", "content": img1})
            conversation_history.append({"type": "text", "content": "COMPLETED WITH CANDIDATE 1:"})

            # Generate final answer
            final_suffix = (
                'Now output EXACTLY ONE <FINAL_ANSWER_JSON>{"choice": 0 or 1, "rationale": "≤30 words"}</FINAL_ANSWER_JSON>\n'
                "Do not output any additional images."
            )
            final_text = self._generate_text_from_conversation(
                conversation_history, final_suffix, text_max_new_tokens, text_do_sample
            )

        else:
            # Maze/Sliding: [gen_text(plan) → gen_image(step)]×k → gen_text(answer)
            eval_logger.info(f"Uni-MMMU {task_type} mode: generating {num_images} steps")

            step_texts = []  # Store all plan texts
            step_images = []  # Store all generated step images

            # Memory optimization: only keep recent images in context during generation
            max_images_in_context = 2  # Only keep last N images to save memory

            for i in range(1, num_images + 1):
                # Generate planning text
                if task_type == "maze":
                    plan_suffix = f'Now planning for step {i}, Please output a sentence in the form: "Next, move one step up/down/left/right."'
                else:  # sliding
                    plan_suffix = f'Now planning for step {i}, Please output a sentence describing which tile to move and in which direction.'

                plan_text = self._generate_text_from_conversation(
                    conversation_history, plan_suffix, 128, text_do_sample
                )
                eval_logger.info(f"Step {i} plan: {plan_text}")
                step_texts.append(plan_text)
                conversation_history.append({"type": "text", "content": plan_text})

                # Generate step image with LIMITED context to save memory
                # Build limited context: input images + recent generated images only
                img_suffix = f"Now, generate the image for step {i}."

                limited_context = []
                # Add input images
                for img in input_images:
                    if img is not None:
                        limited_context.append({"type": "image", "content": img})

                # Add initial prompt
                limited_context.append({"type": "text", "content": prompt})

                # Add only recent step images (last max_images_in_context)
                start_idx = max(0, len(step_images) - max_images_in_context)
                for j in range(start_idx, len(step_images)):
                    limited_context.append({"type": "text", "content": step_texts[j]})
                    limited_context.append({"type": "image", "content": step_images[j]})

                # Add current plan
                limited_context.append({"type": "text", "content": plan_text})

                step_img = self._generate_image_from_conversation(
                    limited_context, img_suffix, cfg_scale, top_k, top_p, h_div_w, t2i_stage
                )
                step_img_path = os.path.join(
                    self.simple_output_dir, "images", f"{task}_{doc_id}_step_{i:04d}.png"
                )
                os.makedirs(os.path.dirname(step_img_path), exist_ok=True)
                step_img.save(step_img_path)
                generated_images.append(step_img_path)
                step_images.append(step_img)
                eval_logger.info(f"Generated step {i} image: {step_img_path}")

                # Don't add image to conversation_history to save memory
                # Just note that we generated an image
                conversation_history.append({"type": "text", "content": f"[Generated step {i} image]"})

            # Rebuild context for final answer generation (align with Uni-MMMU/Bagel)
            eval_logger.info("Rebuilding conversation context for final answer generation")
            conversation_history = []

            # Re-add original input images
            for idx, img in enumerate(input_images):
                if img is not None:
                    conversation_history.append({"type": "image", "content": img})

            # Re-add initial prompt
            conversation_history.append({"type": "text", "content": prompt})

            # Re-add all step texts and images
            for i, (plan_text, step_img) in enumerate(zip(step_texts, step_images), 1):
                conversation_history.append({"type": "text", "content": plan_text})
                conversation_history.append({"type": "text", "content": f"Image for step {i}:"})
                conversation_history.append({"type": "image", "content": step_img})

            # Generate final answer
            final_suffix = (
                "After the images, emit EXACTLY ONE LINE containing ONLY the final move list "
                "as <ANSWER_JSON>[...]</ANSWER_JSON>. No other text."
            )
            final_text = self._generate_text_from_conversation(
                conversation_history, final_suffix, text_max_new_tokens, text_do_sample
            )
            eval_logger.info(f"{task_type} final answer: {final_text}")

        # Unload VAE to free memory
        if hasattr(self._model, 'vae_local') and self._model.vae_local is not None:
            eval_logger.info("Unloading VAE after interleaved generation")
            del self._model.vae_local
            self._model.vae_local = None
            torch.cuda.empty_cache()

        return final_text, generated_images

    def _generate_image_from_prompt(
        self, prompt: str, cfg_scale: float, top_k: int, top_p: float, h_div_w: float, t2i_stage: int
    ) -> Image.Image:
        """
        Generate image from text prompt using OneCAT's generate_t2i.

        Args:
            prompt: Text prompt for image generation
            cfg_scale: Classifier-free guidance scale
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            h_div_w: Height/width ratio
            t2i_stage: T2I stage (2 or 3)

        Returns:
            Generated PIL Image
        """
        try:
            from onecat.constants import SYSTEM_PROMPT

            system_message = SYSTEM_PROMPT
            user_message = f"<|im_start|>user\n{prompt}<|im_end|>"
            assistant_message = "<|im_start|>assistant\n<img_gen>"

            batch = system_message + user_message + assistant_message
            model_inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                max_length=1024,
                truncation=False,
                padding=False,
            )
            input_ids = model_inputs["input_ids"].to(self.device_str)
            attention_mask = model_inputs["attention_mask"].to(self.device_str)

            # CFG batch (empty prompt)
            cfg_batch = system_message + "<|im_start|>user\n<|im_end|><|im_start|>assistant\n<img_gen>"
            model_inputs_cfg = self.tokenizer(
                cfg_batch,
                return_tensors="pt",
                max_length=1024,
                truncation=False,
                padding=False,
            )
            input_ids_cfg = model_inputs_cfg["input_ids"].to(self.device_str)
            attention_mask_cfg = model_inputs_cfg["attention_mask"].to(self.device_str)

            # Generation config
            generation_config = dict(
                output_hidden_states=True,
                cfg=cfg_scale,
                top_k=top_k,
                top_p=top_p,
                use_cache=True,
                return_dict=True,
                h_div_w=h_div_w,
            )

            # Generate image
            with torch.no_grad():
                img = self.model.generate_t2i(
                    input_ids=input_ids,
                    input_ids_cfg=input_ids_cfg,
                    attention_mask=attention_mask,
                    attention_mask_cfg=attention_mask_cfg,
                    **generation_config,
                )

            # Clean up
            del input_ids, input_ids_cfg, attention_mask, attention_mask_cfg
            torch.cuda.empty_cache()

            # Convert to PIL Image
            img_pil = Image.fromarray(
                img[0]
                .add_(1)
                .mul_(0.5)
                .to(dtype=torch.float32)
                .permute(1, 2, 0)
                .mul_(255)
                .cpu()
                .numpy()
                .astype(np.uint8)
            )

            if img_pil.mode != 'RGB':
                img_pil = img_pil.convert('RGB')

            # CRITICAL: Delete the img tensor to free GPU memory
            del img
            torch.cuda.empty_cache()

            return img_pil

        except Exception as e:
            eval_logger.error(f"Image generation failed: {e}")
            raise

    def _generate_image_from_conversation(
        self,
        conversation_history: List[dict],
        suffix: str,
        cfg_scale: float,
        top_k: int,
        top_p: float,
        h_div_w: float,
        t2i_stage: int,
    ) -> Image.Image:
        """
        Generate image from conversation history (including images and text).
        This is similar to _generate_text_from_conversation but for image generation.

        Args:
            conversation_history: List of conversation turns (type: "text" or "image")
            suffix: Additional text to append to prompt
            cfg_scale: Classifier-free guidance scale
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            h_div_w: Height/width ratio
            t2i_stage: T2I stage (2 or 3)

        Returns:
            Generated PIL Image
        """
        try:
            from onecat.constants import SYSTEM_PROMPT, REF_GEN_CONTEXT_TOKEN, REF_GEN_START_TOKEN, REF_GEN_END_TOKEN

            # Setup ref_img_context_token_id if not already set
            if not hasattr(self._model, 'ref_img_context_token_id'):
                ref_img_context_token_id = self.tokenizer.convert_tokens_to_ids(REF_GEN_CONTEXT_TOKEN)
                self._model.ref_img_context_token_id = ref_img_context_token_id
                eval_logger.info(f"Set ref_img_context_token_id: {ref_img_context_token_id}")

            # Collect images and text from conversation history
            images = []
            text_parts = []

            for turn in conversation_history:
                if turn["type"] == "image":
                    images.append(turn["content"])
                elif turn["type"] == "text":
                    text_parts.append(turn["content"])

            # Add suffix
            text_parts.append(suffix)

            # If we have images, use generate_edit; otherwise use generate_t2i
            if len(images) > 0:
                eval_logger.info(f"Generating image with {len(images)} reference images using generate_edit")

                # Process all images and calculate total tokens
                all_pixel_values = []
                total_ref_tokens = 0
                patch_size = self._model.patch_size

                # Resize images to a consistent size
                target_size = 448  # Standard size for reference images

                for idx, img in enumerate(images):
                    # Resize image
                    if img.size != (target_size, target_size):
                        img_resized = img.resize((target_size, target_size), Image.LANCZOS)
                    else:
                        img_resized = img

                    # Load image (this will apply smart_resize internally)
                    pv, _ = self._load_image(img_resized)
                    all_pixel_values.append(pv)

                    # Calculate tokens based on actual pixel_values shape
                    # pv shape: [1, 3, H, W]
                    actual_height = pv.shape[2]
                    actual_width = pv.shape[3]
                    token_width = int(actual_width // patch_size)
                    token_height = int(actual_height // patch_size)
                    num_tokens = int(token_width * token_height * (0.5 ** 2))
                    total_ref_tokens += num_tokens
                    eval_logger.info(f"Image {idx}: pixel_values shape={pv.shape}, actual size={actual_width}x{actual_height}, tokens={num_tokens}")

                # Concatenate all pixel values
                ref_pixel_values = torch.cat(all_pixel_values, dim=0).to(
                    device=self.device_str, dtype=self.torch_dtype
                )
                eval_logger.info(f"Total reference tokens: {total_ref_tokens}, ref_pixel_values shape: {ref_pixel_values.shape}")

                # IMPORTANT: Delete the list to free memory
                del all_pixel_values
                torch.cuda.empty_cache()

                # Build image tokens string
                image_tokens = f'{REF_GEN_START_TOKEN}{REF_GEN_CONTEXT_TOKEN * total_ref_tokens}{REF_GEN_END_TOKEN}'

                # Build full prompt with image tokens at the beginning
                full_text = "\n".join(text_parts)
                full_prompt = image_tokens + "\n" + full_text

                # Build prompt with system message
                system_message = SYSTEM_PROMPT
                user_message = f"<|im_start|>user\n{full_prompt}<|im_end|>"
                assistant_message = "<|im_start|>assistant\n<img_gen>"
                batch = system_message + user_message + assistant_message

                # Tokenize
                model_inputs = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    max_length=2048,
                    truncation=False,
                    padding=False,
                )
                input_ids = model_inputs["input_ids"].to(self.device_str)
                attention_mask = model_inputs["attention_mask"].to(self.device_str)

                # CFG batch (empty text, but keep image tokens)
                cfg_batch = system_message + f"<|im_start|>user\n{image_tokens}<|im_end|><|im_start|>assistant\n<img_gen>"
                model_inputs_cfg = self.tokenizer(
                    cfg_batch,
                    return_tensors="pt",
                    max_length=2048,
                    truncation=False,
                    padding=False,
                )
                input_ids_cfg = model_inputs_cfg["input_ids"].to(self.device_str)
                attention_mask_cfg = model_inputs_cfg["attention_mask"].to(self.device_str)

                # CFG batch 2 (no images, no text)
                cfg_batch2 = system_message + "<|im_start|>user\n<|im_end|><|im_start|>assistant\n<img_gen>"
                model_inputs_cfg2 = self.tokenizer(
                    cfg_batch2,
                    return_tensors="pt",
                    max_length=2048,
                    truncation=False,
                    padding=False,
                )
                input_ids_cfg2 = model_inputs_cfg2["input_ids"].to(self.device_str)
                attention_mask_cfg2 = model_inputs_cfg2["attention_mask"].to(self.device_str)

                # Generation config
                generation_config = dict(
                    output_hidden_states=True,
                    cfg_I=1.5,  # Image CFG scale
                    cfg_T=cfg_scale,  # Text CFG scale
                    top_k=top_k,
                    top_p=top_p,
                    use_cache=True,
                    return_dict=True,
                    h_div_w=h_div_w,
                )

                # Generate image using generate_edit
                with torch.no_grad():
                    img = self.model.generate_edit(
                        pixel_values=ref_pixel_values,
                        input_ids=input_ids,
                        input_ids_cfg=input_ids_cfg,
                        input_ids_cfg2=input_ids_cfg2,
                        attention_mask=attention_mask,
                        attention_mask_cfg=attention_mask_cfg,
                        attention_mask_cfg2=attention_mask_cfg2,
                        **generation_config,
                    )

                # Clean up
                del ref_pixel_values, input_ids, input_ids_cfg, input_ids_cfg2
                del attention_mask, attention_mask_cfg, attention_mask_cfg2
                del model_inputs, model_inputs_cfg, model_inputs_cfg2  # Also delete tokenizer outputs
                torch.cuda.empty_cache()

            else:
                # No images, use text-only generation
                eval_logger.info("Generating image with text-only using generate_t2i")

                full_prompt = "\n".join(text_parts)

                # Build prompt with system message
                system_message = SYSTEM_PROMPT
                user_message = f"<|im_start|>user\n{full_prompt}<|im_end|>"
                assistant_message = "<|im_start|>assistant\n<img_gen>"
                batch = system_message + user_message + assistant_message

                # Tokenize
                model_inputs = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    max_length=1024,
                    truncation=False,
                    padding=False,
                )
                input_ids = model_inputs["input_ids"].to(self.device_str)
                attention_mask = model_inputs["attention_mask"].to(self.device_str)

                # CFG batch (empty prompt)
                cfg_batch = system_message + "<|im_start|>user\n<|im_end|><|im_start|>assistant\n<img_gen>"
                model_inputs_cfg = self.tokenizer(
                    cfg_batch,
                    return_tensors="pt",
                    max_length=1024,
                    truncation=False,
                    padding=False,
                )
                input_ids_cfg = model_inputs_cfg["input_ids"].to(self.device_str)
                attention_mask_cfg = model_inputs_cfg["attention_mask"].to(self.device_str)

                # Generation config
                generation_config = dict(
                    output_hidden_states=True,
                    cfg=cfg_scale,
                    top_k=top_k,
                    top_p=top_p,
                    use_cache=True,
                    return_dict=True,
                    h_div_w=h_div_w,
                )

                # Generate image
                with torch.no_grad():
                    img = self.model.generate_t2i(
                        input_ids=input_ids,
                        input_ids_cfg=input_ids_cfg,
                        attention_mask=attention_mask,
                        attention_mask_cfg=attention_mask_cfg,
                        **generation_config,
                    )

                # Clean up
                del input_ids, input_ids_cfg, attention_mask, attention_mask_cfg
                del model_inputs, model_inputs_cfg  # Also delete tokenizer outputs
                torch.cuda.empty_cache()

            # Convert to PIL Image
            img_pil = Image.fromarray(
                img[0]
                .add_(1)
                .mul_(0.5)
                .to(dtype=torch.float32)
                .permute(1, 2, 0)
                .mul_(255)
                .cpu()
                .numpy()
                .astype(np.uint8)
            )

            if img_pil.mode != 'RGB':
                img_pil = img_pil.convert('RGB')

            # CRITICAL: Delete the img tensor to free GPU memory
            del img
            torch.cuda.empty_cache()

            return img_pil

        except Exception as e:
            eval_logger.error(f"Image generation from conversation failed: {e}")
            raise

    def _generate_text_from_conversation(
        self, conversation_history: List[dict], suffix: str, max_new_tokens: int, do_sample: bool
    ) -> str:
        """
        Generate text response from conversation history.

        Args:
            conversation_history: List of conversation turns (type: "text" or "image")
            suffix: Additional text to append to prompt
            max_new_tokens: Maximum tokens to generate
            do_sample: Whether to use sampling

        Returns:
            Generated text
        """
        try:
            # Build multi-modal prompt from conversation history
            images = []
            text_parts = []

            for turn in conversation_history:
                if turn["type"] == "image":
                    images.append(turn["content"])
                    text_parts.append("<image>")
                elif turn["type"] == "text":
                    text_parts.append(turn["content"])

            # Add suffix
            text_parts.append(suffix)
            full_prompt = "\n".join(text_parts)

            # Process images
            if len(images) > 0:
                # Resize all images to a consistent size before processing
                # Use 448x448 as the standard size for multi-image inputs
                target_size = 448
                resized_images = []

                for img in images:
                    # Resize to square for consistency
                    if img.size != (target_size, target_size):
                        img_resized = img.resize((target_size, target_size), Image.LANCZOS)
                    else:
                        img_resized = img
                    resized_images.append(img_resized)

                # Now process all resized images
                pixel_values_list = []
                pixel_values_thumbnail_list = []

                for img in resized_images:
                    pv, pvt = self._load_image(img)
                    pixel_values_list.append(pv)
                    pixel_values_thumbnail_list.append(pvt)

                pixel_values = torch.cat(pixel_values_list, dim=0).to(
                    device=self.device_str, dtype=self.torch_dtype
                )
                pixel_values_thumbnail = torch.cat(pixel_values_thumbnail_list, dim=0).to(
                    device=self.device_str, dtype=self.torch_dtype
                )

                # Set num_patches_list (1 patch per image)
                num_patches_list = [1] * len(images)
            else:
                pixel_values = None
                pixel_values_thumbnail = None
                num_patches_list = None

            # Generation config
            generation_config = dict(
                do_sample=do_sample,
                top_k=None,
                top_p=None,
                num_beams=1,
                max_new_tokens=max_new_tokens,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.2,  # Add repetition penalty to avoid loops
            )

            # Generate text
            response = self.model.chat(
                tokenizer=self.tokenizer,
                pixel_values=pixel_values,
                question=full_prompt,
                generation_config=generation_config,
                pixel_values_thumbnail=pixel_values_thumbnail,
                num_patches_list=num_patches_list,
                verbose=False,
            )

            # Clean up
            if pixel_values is not None:
                del pixel_values, pixel_values_thumbnail
                torch.cuda.empty_cache()

            return response

        except Exception as e:
            eval_logger.error(f"Text generation failed: {e}")
            raise

    def generate_until(self, requests: List[Instance]) -> List[str]:
        """Main inference method"""
        res = []
        pbar = tqdm(
            total=len(requests),
            disable=(self.rank != 0),
            desc="OneCAT Generating",
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

            # Check if this is Uni-MMMU interleaved generation mode
            # Support both onecat_interleaved and bagel_interleaved for compatibility
            onecat_interleaved = gen_kwargs.get("onecat_interleaved", None)
            if onecat_interleaved is None:
                onecat_interleaved = gen_kwargs.get("bagel_interleaved", None)

            if onecat_interleaved is not None:
                # Uni-MMMU interleaved generation mode
                # Get input images and doc data
                doc = self.task_dict[task][split][doc_id]
                input_images = []
                if doc_to_visual is not None:
                    visuals = [doc_to_visual(doc)]
                    input_images = self.flatten(visuals)

                output_text, output_images = self.generate_uni_mmmu_interleaved(
                    input_images, prompt, str(doc_id), task, onecat_interleaved, doc
                )
                formatted_output = self.format_output(output_text, output_images)

            else:
                # Standard understanding mode
                # Get image from doc_to_visual
                if doc_to_visual is None:
                    eval_logger.warning(
                        f"No image provided for understanding mode, doc_id={doc_id}"
                    )
                    res.append("")
                    pbar.update(1)
                    continue

                # Get image from doc_to_visual
                visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
                visuals = self.flatten(visuals)

                if not visuals or len(visuals) == 0:
                    eval_logger.warning(f"No visual data found for doc_id={doc_id}")
                    res.append("")
                    pbar.update(1)
                    continue

                # Use first image for understanding
                image = visuals[0]
                output_text = self.understand_image(prompt, image)
                formatted_output = output_text

            res.append(formatted_output)

            # Save simple output
            if self.simple_output and onecat_interleaved is None:
                result_entry = {
                    "doc_id": str(doc_id),
                    "task": task,
                    "split": split,
                    "mode": "understanding",
                    "prompt": prompt,
                    "output": output_text
                }

                # Save input image (convert RGBA to RGB if needed for JPEG)
                image_filename = f"{doc_id}.jpg"
                image_path = os.path.join(self.simple_output_dir, "images", image_filename)
                if image.mode == 'RGBA':
                    image = image.convert('RGB')
                image.save(image_path)
                result_entry["input_image"] = f"./images/{image_filename}"

                self.simple_results.append(result_entry)

                # Save results to JSON file
                results_file = os.path.join(self.simple_output_dir, "results.json")
                with open(results_file, "w", encoding="utf-8") as f:
                    json.dump(self.simple_results, f, ensure_ascii=False, indent=2)

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
            "OneCAT is a generation model and does not support loglikelihood"
        )

    def generate_until_multi_round(self, requests) -> List[str]:
        """Multi-round dialogue generation"""
        raise NotImplementedError(
            "TODO: Implement multi-round dialogue generation"
        )
