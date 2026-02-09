"""
OneCAT Visual Chain-of-Thought Model

Two-stage inference:
1. Stage 1: Generate visualization image from text prompt
2. Stage 2: Answer question using the generated image and original image

Usage:
    python -m lmms_eval \
        --model onecat_visual_cot \
        --model_args pretrained=/path/to/OneCAT-3B,vae_path=/path/to/infinity_vae_d32reg.pth \
        --tasks illusionbench_arshia_icon_shape_visual_cot \
        --batch_size 1 \
        --output_path ./logs/
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
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


@register_model("onecat_visual_cot")
class OneCATVisualCoT(lmms):
    """
    OneCAT Visual Chain-of-Thought Model

    Performs two-stage visual reasoning:
    1. Generate visualization image from text prompt
    2. Answer question using the generated image and original image
    """

    def __init__(
        self,
        pretrained: str,
        vae_path: str,
        # Stage 1: Image generation parameters
        stage1_t2i_stage: int = 3,
        stage1_h_div_w: float = 1.0,
        stage1_cfg: float = 20.0,
        stage1_top_k: int = 2,
        stage1_top_p: float = 0.97,
        stage1_max_input_tokens: int = 1024,
        stage1_ref_image_size: int = 672,  # Max size for reference image in generate_edit (lower = less memory)
        stage1_cfg_I: float = 1.0,  # Image CFG strength for generate_edit
        stage1_cfg_T: float = 3.0,  # Text CFG strength for generate_edit
        # Stage 2: Visual understanding parameters
        stage2_max_new_tokens: int = 512,
        stage2_do_sample: bool = False,
        stage2_num_beams: int = 1,
        stage2_top_k: Optional[int] = None,
        stage2_top_p: Optional[float] = None,
        # Generation prompt template
        generation_prompt_template: str = "Generate a detailed visual diagram or illustration to help answer this question: {question}",
        # Output and debugging
        output_dir: Optional[str] = None,
        save_intermediate: bool = False,
        intermediate_dir: Optional[str] = None,
        simple_output: bool = True,
        simple_output_dir: Optional[str] = None,
        # Error handling
        fail_gracefully: bool = True,
        # Model loading
        dtype: str = "bfloat16",
        device: str = "cuda",
        seed: int = 0,
        **kwargs,
    ) -> None:
        super().__init__()

        self.pretrained = pretrained
        self.vae_path = vae_path
        self.save_intermediate = save_intermediate
        self.fail_gracefully = fail_gracefully
        self.seed = seed
        self.generation_prompt_template = generation_prompt_template
        self.device_str = device

        # Stage 1 parameters
        self.stage1_t2i_stage = stage1_t2i_stage
        self.stage1_h_div_w = stage1_h_div_w
        self.stage1_cfg = stage1_cfg
        self.stage1_top_k = stage1_top_k
        self.stage1_top_p = stage1_top_p
        self.stage1_max_input_tokens = stage1_max_input_tokens
        self.stage1_ref_image_size = stage1_ref_image_size
        self.stage1_cfg_I = stage1_cfg_I
        self.stage1_cfg_T = stage1_cfg_T

        # Stage 2 parameters
        # Handle kwargs that might override defaults (e.g., from command line)
        if "max_new_tokens" in kwargs:
            stage2_max_new_tokens = kwargs.pop("max_new_tokens")
            eval_logger.info(f"Overriding stage2_max_new_tokens with {stage2_max_new_tokens} from kwargs")
        self.stage2_max_new_tokens = stage2_max_new_tokens
        self.stage2_do_sample = stage2_do_sample
        self.stage2_num_beams = stage2_num_beams
        self.stage2_top_k = stage2_top_k
        self.stage2_top_p = stage2_top_p
        
        # Log any remaining kwargs
        if kwargs:
            eval_logger.warning(f"Unused kwargs: {kwargs}")

        # Determine dtype
        if dtype == "bfloat16":
            self.torch_dtype = torch.bfloat16
        elif dtype == "float16":
            self.torch_dtype = torch.float16
        else:
            self.torch_dtype = torch.float32

        # Setup output directories
        if output_dir is None:
            self.output_dir = "./logs/onecat_visual_cot"
        else:
            self.output_dir = output_dir

        if intermediate_dir is None:
            self.intermediate_dir = self.output_dir
        else:
            self.intermediate_dir = intermediate_dir

        if save_intermediate:
            os.makedirs(self.intermediate_dir, exist_ok=True)
            eval_logger.info(
                f"Intermediate artifacts will be saved under: {self.intermediate_dir}"
            )

        # Setup simple output
        self.simple_output = simple_output
        if simple_output_dir is None:
            self.simple_output_dir = "./logs/onecat_cot_simple_output"
        else:
            self.simple_output_dir = simple_output_dir

        if self.simple_output:
            os.makedirs(self.simple_output_dir, exist_ok=True)
            os.makedirs(os.path.join(self.simple_output_dir, "generated_images"), exist_ok=True)
            self.simple_results = []

        # Setup rank and world size
        self._rank = 0
        self._world_size = 1

        # Load models
        eval_logger.info(f"Loading OneCAT model from {pretrained}")
        self._load_model()

        eval_logger.info("OneCATVisualCoT initialized successfully")

    def _load_model(self):
        """Load OneCAT model, tokenizer, and VAE"""
        try:
            from onecat.constants import (
                IMG_GEN_CONTEXT_TOKEN,
                IMG_GEN_START_TOKEN,
                SYSTEM_PROMPT,
            )
            from onecat.modeling_onecat import OneCatVLModel
            from onecat.smart_resize import smart_resize
            from onecat.util import build_transform
            from onecat.var_model.tools.run_infinity import load_visual_tokenizer

            self.IMG_GEN_CONTEXT_TOKEN = IMG_GEN_CONTEXT_TOKEN
            self.IMG_GEN_START_TOKEN = IMG_GEN_START_TOKEN
            self.SYSTEM_PROMPT = SYSTEM_PROMPT
            self.OneCatVLModel = OneCatVLModel
            self.smart_resize = smart_resize
            self.build_transform = build_transform
            self.load_visual_tokenizer = load_visual_tokenizer

        except Exception as e:
            raise ImportError(
                f"Failed to import OneCAT dependencies. "
                f"Please ensure:\n"
                f"  1. OneCAT repository is available at {onecat_path}\n"
                f"  2. Required dependencies are installed\n"
                f"Error: {e}"
            )

        # Load model
        # Note: device_map="auto" can cause dtype issues, so we load normally first
        eval_logger.info(f"Loading model from {self.pretrained}")
        self._model = self.OneCatVLModel.from_pretrained(self.pretrained)

        # Move model to device with correct dtype
        self._model = self._model.to(
            device=self.device_str, dtype=self.torch_dtype
        ).eval()

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self.pretrained)

        # Store VAE args for lazy loading (don't load VAE yet)
        eval_logger.info(f"Preparing VAE config (will load on-demand)")
        if self.stage1_t2i_stage == 2:
            pn = "0.25M"
        elif self.stage1_t2i_stage == 3:
            pn = "1M"
        else:
            raise ValueError(
                f"Expected t2i_stage 2 or 3. Got {self.stage1_t2i_stage}"
            )

        self.vae_args = argparse.Namespace(
            vae_type=32,
            vae_path=self.vae_path,
            apply_spatial_patchify=0,
            pn=pn,
        )

        # Setup token IDs
        img_gen_context_token_id = self.tokenizer.convert_tokens_to_ids(
            self.IMG_GEN_CONTEXT_TOKEN
        )
        img_gen_start_token_id = self.tokenizer.convert_tokens_to_ids(
            self.IMG_GEN_START_TOKEN
        )
        self._model.img_gen_context_token_id = img_gen_context_token_id
        self._model.img_gen_start_token_id = img_gen_start_token_id

        # Setup reference image token for generate_edit
        try:
            from onecat.constants import REF_GEN_CONTEXT_TOKEN
            ref_img_context_token_id = self.tokenizer.convert_tokens_to_ids(
                REF_GEN_CONTEXT_TOKEN
            )
            self._model.ref_img_context_token_id = ref_img_context_token_id
            eval_logger.info("Reference image token ID configured for generate_edit")
        except ImportError:
            eval_logger.warning("Could not import REF_GEN_CONTEXT_TOKEN, generate_edit may not work")

        eval_logger.info(
            f"Model loaded with dtype={self.torch_dtype}, device={self.device_str}"
        )
        eval_logger.info("💡 VAE will be loaded on-demand in Stage 1 and unloaded before Stage 2")

    def _load_vae(self):
        """Load VAE on-demand for Stage 1"""
        if not hasattr(self._model, 'vae_local') or self._model.vae_local is None:
            eval_logger.info(f"Loading Infinity VAE from {self.vae_path} (~1.5GB)")
            vae_local = self.load_visual_tokenizer(self.vae_args)
            vae_local.eval()
            # Move VAE to same device and dtype as main model
            vae_local = vae_local.to(device=self.device_str, dtype=self.torch_dtype)
            self._model.vae_local = vae_local
            self._model.vargpt_gen_args = self.vae_args
            eval_logger.info(f"VAE loaded to {self.device_str} with dtype {self.torch_dtype}")
            eval_logger.info(f"✅ VAE will be unloaded after Stage 1 to free ~0.75-1.5GB GPU memory")

    def _unload_vae(self):
        """Unload VAE to free GPU memory after Stage 1"""
        if hasattr(self._model, 'vae_local') and self._model.vae_local is not None:
            eval_logger.info("Unloading VAE to free ~0.75-1.5GB GPU memory")
            del self._model.vae_local
            self._model.vae_local = None

            # Clear model's KV cache if it exists
            if hasattr(self._model, 'past_key_values'):
                self._model.past_key_values = None

            # Clear any cached outputs in the model
            if hasattr(self._model, '_cache'):
                self._model._cache = None

            torch.cuda.empty_cache()

            # Force garbage collection
            import gc
            gc.collect()

            eval_logger.info("VAE unloaded successfully")

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
        """Load and preprocess image for OneCAT model"""
        width, height = image.size
        eval_logger.info(f"_load_image - Input image size: {width}x{height}")

        # Smart resize
        resized_height, resized_width = self.smart_resize(height, width)
        eval_logger.info(f"_load_image - After smart_resize: {resized_width}x{resized_height}")
        transform = self.build_transform(input_size=(resized_height, resized_width))
        pixel_values = transform(image).unsqueeze(0)
        eval_logger.info(f"_load_image - pixel_values shape: {pixel_values.shape}")

        # Thumbnail (base size 448x448)
        transform_base = self.build_transform(input_size=(448, 448))
        pixel_values_thumbnail = transform_base(image).unsqueeze(0)
        eval_logger.info(f"_load_image - pixel_values_thumbnail shape: {pixel_values_thumbnail.shape}")

        return pixel_values, pixel_values_thumbnail

    def _center_crop_to_ratio(self, image: Image.Image, h_div_w: float) -> Image.Image:
        """
        Center crop the image to a target height/width ratio (h_div_w).
        Returns a new PIL.Image object.
        """
        width, height = image.size
        target_ratio = h_div_w

        if width * target_ratio > height:
            new_width = int(height / target_ratio)
            new_height = height
        else:
            new_width = width
            new_height = int(width * target_ratio)

        left = (width - new_width) // 2
        top = (height - new_height) // 2
        right = left + new_width
        bottom = top + new_height

        return image.crop((left, top, right, bottom))

    def _stage1_generate_image(
        self, generation_prompt: str, doc_id: str, task: str, original_image: Optional[Image.Image] = None
    ) -> List[str]:
        """
        Stage 1: Generate visualization image from prompt and original image

        Args:
            generation_prompt: Text prompt for image generation
            doc_id: Document ID for file naming
            task: Task name for file naming
            original_image: Original image to use as reference (optional)

        Returns:
            List of image paths
        """
        eval_logger.debug(f"Stage 1 - Generating image for doc {doc_id}")
        eval_logger.debug(f"Generation prompt: {generation_prompt}")

        # Load VAE on-demand for Stage 1
        self._load_vae()

        # Debug: Check if original image is provided
        if original_image is not None:
            eval_logger.info(f"✅ Stage 1 - Original image PROVIDED: size={original_image.size}, mode={original_image.mode}")
        else:
            eval_logger.warning(f"❌ Stage 1 - Original image NOT PROVIDED (will use T2I fallback)")

        try:
            # Check if we have an original image to use as reference
            if original_image is not None:
                eval_logger.info("=" * 80)
                eval_logger.info("Stage 1 - Using generate_edit() with original image as reference")
                eval_logger.info("=" * 80)

                # Process reference image
                ref_image = original_image.convert('RGB')
                h_div_w = ref_image.size[1] / ref_image.size[0]
                width, height = ref_image.size
                eval_logger.info(f"Stage 1 - Original image size: {width}x{height}, h_div_w={h_div_w:.3f}")

                # Use configurable reference image size (default 448 to save memory)
                # This is much smaller than the original 672/1008 to avoid OOM
                short_side_ref = self.stage1_ref_image_size
                eval_logger.info(f"Stage 1 - Using reference image size: {short_side_ref}")

                # Import h_div_w_templates for aspect ratio matching
                try:
                    from onecat.var_model.infinity.utils.dynamic_resolution import h_div_w_templates
                    h_div_w_template = h_div_w_templates[np.argmin(np.abs(float(h_div_w) - h_div_w_templates))]
                    eval_logger.info(f"Stage 1 - h_div_w={h_div_w:.3f}, matched h_div_w_template={h_div_w_template:.3f}")
                except ImportError:
                    eval_logger.warning("Could not import h_div_w_templates, using original h_div_w")
                    h_div_w_template = h_div_w

                # IMPORTANT: Center crop using ORIGINAL h_div_w (not template!)
                # This matches the official generate_imgedit.py implementation
                ref_image = self._center_crop_to_ratio(ref_image, h_div_w)
                width, height = ref_image.size
                eval_logger.info(f"Stage 1 - After center crop to h_div_w={h_div_w:.3f}: {width}x{height}")

                # Resize image using ORIGINAL h_div_w (not template!)
                # IMPORTANT: Limit the maximum dimension to avoid OOM
                max_ref_size = short_side_ref * 1.5  # Allow 1.5x the short side as max
                if width < height:
                    target_h = int(short_side_ref * h_div_w)
                    target_w = short_side_ref
                else:
                    target_h = short_side_ref
                    target_w = int(short_side_ref / h_div_w)

                # Clamp to max size to prevent OOM
                if target_w > max_ref_size:
                    scale = max_ref_size / target_w
                    target_w = int(max_ref_size)
                    target_h = int(target_h * scale)
                    eval_logger.warning(f"Stage 1 - Clamping width from {int(short_side_ref / h_div_w)} to {target_w} to prevent OOM")

                if target_h > max_ref_size:
                    scale = max_ref_size / target_h
                    target_h = int(max_ref_size)
                    target_w = int(target_w * scale)
                    eval_logger.warning(f"Stage 1 - Clamping height from {int(short_side_ref * h_div_w)} to {target_h} to prevent OOM")

                h_bar, w_bar = self.smart_resize(target_h, target_w)
                width, height = [w_bar, h_bar]
                eval_logger.info(f"Stage 1 - After smart_resize: {width}x{height} (clamped to max {max_ref_size})")
                ref_transform = self.build_transform(input_size=[height, width])

                # Calculate token dimensions
                patch_size = self.model.patch_size
                token_width = int(width // patch_size)
                token_height = int(height // patch_size)
                num_ref_image_token = int(token_width * token_height * (0.5 ** 2))
                eval_logger.info(f"Stage 1 - patch_size={patch_size}, token_dims={token_width}x{token_height}, num_ref_image_token={num_ref_image_token}")

                ref_pixel_values = ref_transform(ref_image).unsqueeze(0).to(
                    device=self.device_str, dtype=self.torch_dtype
                )
                eval_logger.info(f"Stage 1 - ref_pixel_values shape: {ref_pixel_values.shape}")

                # Import reference generation tokens
                try:
                    from onecat.constants import REF_GEN_CONTEXT_TOKEN, REF_GEN_START_TOKEN, REF_GEN_END_TOKEN
                except ImportError:
                    eval_logger.error("Could not import reference generation tokens from onecat.constants")
                    raise

                # Prepare text input with reference image tokens
                image_tokens = f'{REF_GEN_START_TOKEN}{REF_GEN_CONTEXT_TOKEN * num_ref_image_token}{REF_GEN_END_TOKEN}'
                prompt_with_image = image_tokens + generation_prompt
                eval_logger.info(f"Stage 1 - Image tokens length: {len(image_tokens)}, total prompt length: {len(prompt_with_image)}")

                system_message = self.SYSTEM_PROMPT
                user_message = f'<|im_start|>user\n{prompt_with_image}<|im_end|>'
                assistant_message = '<|im_start|>assistant\n<img_gen>'
                batch = system_message + user_message + assistant_message

                model_inputs = self.tokenizer(
                    batch,
                    return_tensors='pt',
                    max_length=self.stage1_max_input_tokens,
                    truncation=False,
                    padding=False
                )
                input_ids = model_inputs['input_ids'].to(self.device_str)
                attention_mask = model_inputs['attention_mask'].to(self.device_str)

                # CFG inputs (text only, no image)
                cfg_batch = system_message + f'<|im_start|>user\n{generation_prompt}<|im_end|><|im_start|>assistant\n<img_gen>'
                model_inputs_cfg = self.tokenizer(
                    cfg_batch,
                    return_tensors='pt',
                    max_length=self.stage1_max_input_tokens,
                    truncation=False,
                    padding=False
                )
                input_ids_cfg = model_inputs_cfg['input_ids'].to(self.device_str)
                attention_mask_cfg = model_inputs_cfg['attention_mask'].to(self.device_str)

                # Second CFG input (no text, no image)
                cfg_batch2 = system_message + f'<|im_start|>user<|im_end|><|im_start|>assistant\n<img_gen>'
                model_inputs_cfg2 = self.tokenizer(
                    cfg_batch2,
                    return_tensors='pt',
                    max_length=self.stage1_max_input_tokens,
                    truncation=False,
                    padding=False
                )
                input_ids_cfg2 = model_inputs_cfg2['input_ids'].to(self.device_str)
                attention_mask_cfg2 = model_inputs_cfg2['attention_mask'].to(self.device_str)

                # Generation config for image editing
                generation_config = dict(
                    output_hidden_states=True,
                    cfg_I=self.stage1_cfg_I,  # Image CFG strength
                    cfg_T=self.stage1_cfg_T,  # Text CFG strength
                    top_k=self.stage1_top_k,
                    top_p=self.stage1_top_p,
                    h_div_w=h_div_w_template,
                    use_cache=True,
                    return_dict=True,
                )

                eval_logger.info(f"Stage 1 - Generation config: cfg_I={self.stage1_cfg_I}, cfg_T={self.stage1_cfg_T}, h_div_w={h_div_w_template}")

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

                # Clean up intermediate tensors to free GPU memory
                del ref_pixel_values, input_ids, input_ids_cfg, input_ids_cfg2
                del attention_mask, attention_mask_cfg, attention_mask_cfg2
                # Also delete the model inputs to free memory
                del model_inputs, model_inputs_cfg, model_inputs_cfg2
                torch.cuda.empty_cache()
                eval_logger.info("Stage 1 - Cleaned up intermediate tensors")

            else:
                eval_logger.info("=" * 80)
                eval_logger.info("Stage 1 - Using generate_t2i() without reference image (T2I fallback)")
                eval_logger.warning("⚠️  No original image provided, using text-only generation")
                eval_logger.info("=" * 80)

                # Original text-to-image generation (fallback when no image available)
                system_message = self.SYSTEM_PROMPT
                user_message = f"<|im_start|>user\n{generation_prompt}<|im_end|>"
                assistant_message = "<|im_start|>assistant\n<img_gen>"

                batch = system_message + user_message + assistant_message
                model_inputs = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    max_length=self.stage1_max_input_tokens,
                    truncation=False,
                    padding=False,
                )
                input_ids = model_inputs["input_ids"].to(self.device_str)
                attention_mask = model_inputs["attention_mask"].to(self.device_str)

                # CFG batch (empty prompt for classifier-free guidance)
                cfg_batch = (
                    system_message
                    + "<|im_start|>user\n<|im_end|><|im_start|>assistant\n<img_gen>"
                )
                model_inputs_cfg = self.tokenizer(
                    cfg_batch,
                    return_tensors="pt",
                    max_length=self.stage1_max_input_tokens,
                    truncation=False,
                    padding=False,
                )
                input_ids_cfg = model_inputs_cfg["input_ids"].to(self.device_str)
                attention_mask_cfg = model_inputs_cfg["attention_mask"].to(
                    self.device_str
                )

                # Generation config
                generation_config = dict(
                    output_hidden_states=True,
                    cfg=self.stage1_cfg,
                    top_k=self.stage1_top_k,
                    top_p=self.stage1_top_p,
                    use_cache=True,
                    return_dict=True,
                    h_div_w=self.stage1_h_div_w,
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

                # Clean up intermediate tensors to free GPU memory
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

            # Clean up the img tensor immediately
            del img
            torch.cuda.empty_cache()

            # Ensure image is in RGB mode
            if img_pil.mode != 'RGB':
                eval_logger.warning(f"Converting generated image from {img_pil.mode} to RGB")
                img_pil = img_pil.convert('RGB')
            
            # Ensure generated image is large enough (minimum 448x448)
            min_size = 448
            width, height = img_pil.size
            eval_logger.info(f"Stage 1 - Generated image size: {width}x{height}")
            
            if width < min_size or height < min_size:
                eval_logger.warning(
                    f"Generated image too small ({width}x{height}), resizing to minimum {min_size}x{min_size}"
                )
                # Resize while maintaining aspect ratio
                if width < height:
                    new_width = min_size
                    new_height = int(height * min_size / width)
                else:
                    new_height = min_size
                    new_width = int(width * min_size / height)
                # Ensure both dimensions are at least min_size
                if new_height < min_size:
                    new_height = min_size
                if new_width < min_size:
                    new_width = min_size
                img_pil = img_pil.resize((new_width, new_height), Image.LANCZOS)
                eval_logger.info(f"Resized generated image to {new_width}x{new_height}")

            # Save image
            artifact_dir = os.path.join(self.intermediate_dir, task)
            os.makedirs(artifact_dir, exist_ok=True)
            image_path = os.path.join(
                artifact_dir, f"{doc_id}_stage1_generated.png"
            )
            img_pil.save(image_path)

            eval_logger.debug(f"Stage 1 - Generated image saved to {image_path}")
            return [image_path]

        except Exception as e:
            eval_logger.error(f"Stage 1 failed for doc {doc_id}: {e}")
            if self.fail_gracefully:
                return []
            else:
                raise

    def _stage2_answer_with_image(
        self,
        question: str,
        image_path: str,
        doc_id: str,
        original_image: Optional[Image.Image] = None,
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
        """
        eval_logger.debug(f"Stage 2 - Answering question for doc {doc_id}")
        eval_logger.debug(f"Question: {question}")

        # Debug: Check inputs
        eval_logger.info(f"Stage 2 - Auxiliary image path: {image_path}")
        if original_image is not None:
            eval_logger.info(f"✅ Stage 2 - Original image PROVIDED: size={original_image.size}, mode={original_image.mode}")
        else:
            eval_logger.warning(f"❌ Stage 2 - Original image NOT PROVIDED (will use auxiliary image only)")

        try:
            # Load generated auxiliary image
            auxiliary_image = Image.open(image_path).convert("RGB")

            # Check and resize image if too small
            # OneCAT uses patch_size=14, downsample_ratio=0.5
            # So final tokens = ((image_size / 14) / 2)
            # To ensure at least 16x16 tokens, need image >= 14*2*16 = 448
            min_size = 448
            width, height = auxiliary_image.size
            if width < min_size or height < min_size:
                eval_logger.warning(
                    f"Auxiliary image too small ({width}x{height}), resizing to minimum {min_size}x{min_size}"
                )
                # Resize while maintaining aspect ratio
                if width < height:
                    new_width = min_size
                    new_height = int(height * min_size / width)
                else:
                    new_height = min_size
                    new_width = int(width * min_size / height)
                # Ensure both dimensions are at least min_size
                if new_height < min_size:
                    new_height = min_size
                if new_width < min_size:
                    new_width = min_size
                auxiliary_image = auxiliary_image.resize((new_width, new_height), Image.LANCZOS)
                eval_logger.info(f"Resized auxiliary image to {new_width}x{new_height}")

            # For OneCAT Visual CoT: pass BOTH original image and auxiliary image as separate images
            if original_image is not None:
                eval_logger.info(
                    "Stage 2 - Using multiple images: original image + auxiliary image"
                )

                # Resize both images to the same size before processing
                # Use a standard size (e.g., 448x448) to ensure they can be concatenated
                target_size = 448

                # Resize original image
                orig_w, orig_h = original_image.size
                if orig_w != target_size or orig_h != target_size:
                    # Resize to square for simplicity
                    original_resized = original_image.resize((target_size, target_size), Image.LANCZOS)
                    eval_logger.info(f"Stage 2 - Resized original image from {orig_w}x{orig_h} to {target_size}x{target_size}")
                else:
                    original_resized = original_image

                # Resize auxiliary image
                aux_w, aux_h = auxiliary_image.size
                if aux_w != target_size or aux_h != target_size:
                    auxiliary_resized = auxiliary_image.resize((target_size, target_size), Image.LANCZOS)
                    eval_logger.info(f"Stage 2 - Resized auxiliary image from {aux_w}x{aux_h} to {target_size}x{target_size}")
                else:
                    auxiliary_resized = auxiliary_image

                # Process both images separately (now they have the same size)
                # Load original image
                orig_pixel_values, orig_pixel_values_thumbnail = self._load_image(original_resized)

                # Load auxiliary image
                aux_pixel_values, aux_pixel_values_thumbnail = self._load_image(auxiliary_resized)

                eval_logger.info(f"Stage 2 - orig_pixel_values shape: {orig_pixel_values.shape}")
                eval_logger.info(f"Stage 2 - aux_pixel_values shape: {aux_pixel_values.shape}")

                # Concatenate along batch dimension (dim=0)
                pixel_values = torch.cat([orig_pixel_values, aux_pixel_values], dim=0)
                pixel_values_thumbnail = torch.cat([orig_pixel_values_thumbnail, aux_pixel_values_thumbnail], dim=0)

                pixel_values = pixel_values.to(
                    device=self.device_str, dtype=self.torch_dtype
                )
                pixel_values_thumbnail = pixel_values_thumbnail.to(
                    device=self.device_str, dtype=self.torch_dtype
                )

                eval_logger.info(f"Stage 2 - pixel_values shape: {pixel_values.shape}")
                eval_logger.info(f"Stage 2 - pixel_values_thumbnail shape: {pixel_values_thumbnail.shape}")

                # Use two <image> tokens for two images
                if '<image>' not in question:
                    enhanced_question = '<image>\n<image>\n' + question
                else:
                    # Count existing <image> tokens
                    num_existing_images = question.count('<image>')
                    if num_existing_images == 1:
                        # Add one more <image> token for auxiliary image
                        enhanced_question = '<image>\n' + question
                    else:
                        # Already has multiple images or correct format
                        enhanced_question = question

                # Set num_patches_list for two images (each image is 1 patch)
                num_patches_list = [1, 1]

                eval_logger.info(f"Stage 2 - Using {len(num_patches_list)} images with num_patches_list={num_patches_list}")

            else:
                eval_logger.debug("Stage 2 - Using auxiliary image only")
                pixel_values, pixel_values_thumbnail = self._load_image(
                    auxiliary_image
                )
                pixel_values = pixel_values.to(
                    device=self.device_str, dtype=self.torch_dtype
                )
                pixel_values_thumbnail = pixel_values_thumbnail.to(
                    device=self.device_str, dtype=self.torch_dtype
                )

                # Single image, add one <image> token if missing
                if '<image>' not in question:
                    enhanced_question = '<image>\n' + question
                else:
                    enhanced_question = question

                # Single image
                num_patches_list = [1]

            # Log image tensor shapes for debugging
            num_images = pixel_values.shape[0]
            eval_logger.info(f"Stage 2 - Number of images in batch: {num_images}")
            eval_logger.info(f"Stage 2 - pixel_values shape: {pixel_values.shape}")
            eval_logger.info(f"Stage 2 - pixel_values_thumbnail shape: {pixel_values_thumbnail.shape}")

            # Calculate expected tokens for validation
            patch_size = 14  # OneCAT default
            expected_token_h = int((pixel_values.shape[-2] // patch_size) // 2)
            expected_token_w = int((pixel_values.shape[-1] // patch_size) // 2)
            expected_tokens = expected_token_h * expected_token_w
            eval_logger.info(f"Stage 2 - Expected tokens per image: {expected_tokens} ({expected_token_h}x{expected_token_w})")

            if expected_tokens == 0:
                error_msg = (
                    f"Image too small! pixel_values shape: {pixel_values.shape}, "
                    f"resulting in 0 tokens. Need at least 448x448 image."
                )
                eval_logger.error(error_msg)
                if self.fail_gracefully:
                    return ""
                else:
                    raise ValueError(error_msg)

            # Generation config
            generation_config = dict(
                do_sample=self.stage2_do_sample,
                top_k=self.stage2_top_k,
                top_p=self.stage2_top_p,
                num_beams=self.stage2_num_beams,
                max_new_tokens=self.stage2_max_new_tokens,
                eos_token_id=self.tokenizer.eos_token_id,
            )

            # Debug: Log generation config
            eval_logger.info(f"Stage 2 - Generation config: {generation_config}")
            eval_logger.info(f"Stage 2 - Question length: {len(enhanced_question)}")
            eval_logger.info(f"Stage 2 - num_patches_list: {num_patches_list}")

            # Validate question is not empty
            if not enhanced_question or len(enhanced_question.strip()) == 0:
                error_msg = f"Enhanced question is empty! Original question: {question}"
                eval_logger.error(error_msg)
                if self.fail_gracefully:
                    return ""
                else:
                    raise ValueError(error_msg)

            eval_logger.info(f"Stage 2 - About to call model.chat")

            # Generate answer with num_patches_list for multiple images
            response = self.model.chat(
                tokenizer=self.tokenizer,
                pixel_values=pixel_values,
                question=enhanced_question,
                generation_config=generation_config,
                pixel_values_thumbnail=pixel_values_thumbnail,
                num_patches_list=num_patches_list,
                verbose=True,  # Enable verbose for debugging
            )

            # Clean up GPU memory after generation
            del pixel_values, pixel_values_thumbnail
            torch.cuda.empty_cache()

            eval_logger.debug(f"Stage 2 - Generated answer: {response[:100]}...")
            return response

        except Exception as e:
            import traceback
            eval_logger.error(f"Stage 2 failed for doc {doc_id}: {type(e).__name__}: {e}")
            eval_logger.error(f"Full traceback:\n{traceback.format_exc()}")
            if self.fail_gracefully:
                return ""
            else:
                raise

    def _save_intermediate_artifacts(
        self,
        doc_id: str,
        task: str,
        generation_prompt: str,
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
            "generated_images": generated_images,
            "question": question,
            "stage2_answer": stage2_answer,
        }

        metadata_path = os.path.join(artifact_dir, f"{doc_id}_metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        eval_logger.debug(f"Saved intermediate artifacts to: {metadata_path}")

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
            desc="OneCATVisualCoT Generating",
        )

        for request in requests:
            contexts, gen_kwargs, doc_to_visual, doc_id, task, split = request.args

            # Extract original image from document using task_dict
            original_image = None
            if doc_to_visual is not None:
                try:
                    # Get doc from task_dict
                    doc = self.task_dict[task][split][doc_id]
                    original_visuals = doc_to_visual(doc)
                    if original_visuals and len(original_visuals) > 0:
                        original_image = original_visuals[0]
                        eval_logger.debug(
                            f"Extracted original image for doc {doc_id}"
                        )
                except Exception as e:
                    eval_logger.warning(
                        f"Failed to extract original image for doc {doc_id}: {e}"
                    )

            # Parse contexts to extract generation_prompt if provided
            gen_prompt_match = re.search(
                r"\[GEN_PROMPT\](.*?)\[/GEN_PROMPT\]", contexts, re.DOTALL
            )
            question_match = re.search(
                r"\[QUESTION\](.*?)\[/QUESTION\]", contexts, re.DOTALL
            )

            if gen_prompt_match and question_match:
                # Use custom generation prompt from task config
                custom_gen_prompt = gen_prompt_match.group(1).strip()
                actual_question = question_match.group(1).strip()
                generation_prompt = custom_gen_prompt.replace(
                    "{question}", actual_question
                )
                # Extract just the question for stage 2
                contexts = actual_question
                eval_logger.info("Using custom generation prompt from task config")
            else:
                # Use default template
                generation_prompt = self.generation_prompt_template.format(
                    question=contexts
                )

            eval_logger.info(f"\n{'='*60}")
            eval_logger.info(f"Processing doc {doc_id} from task {task}")
            eval_logger.info(f"{'='*60}")

            # Stage 1: Generate visualization image (with original image if available)
            generated_images = self._stage1_generate_image(
                generation_prompt=generation_prompt,
                doc_id=doc_id,
                task=task,
                original_image=original_image
            )

            # Unload VAE after Stage 1 to free GPU memory (~0.75-1.5GB)
            self._unload_vae()

            # Check if image was generated
            if not generated_images or len(generated_images) == 0:
                eval_logger.warning(
                    f"No image generated for doc {doc_id}, returning empty answer"
                )
                res.append("")
                pbar.update(1)
                continue

            # Stage 2: Answer question using generated image (and original image if available)
            final_answer = self._stage2_answer_with_image(
                question=contexts,
                image_path=generated_images[0],
                doc_id=doc_id,
                original_image=original_image,
            )

            # Save intermediate artifacts if enabled
            self._save_intermediate_artifacts(
                doc_id=doc_id,
                task=task,
                generation_prompt=generation_prompt,
                generated_images=generated_images,
                question=contexts,
                stage2_answer=final_answer,
            )

            # Save simple output if enabled
            if self.simple_output:
                result_entry = {
                    "doc_id": str(doc_id),
                    "task": task,
                    "split": split,
                    "mode": "visual_cot",
                    "generation_prompt": generation_prompt,
                    "question": contexts,
                    "output": final_answer,
                }

                # Copy generated image to simple output directory
                if generated_images and len(generated_images) > 0:
                    import shutil

                    gen_image_filename = f"{doc_id}_generated.jpg"
                    gen_image_dest = os.path.join(
                        self.simple_output_dir, "generated_images", gen_image_filename
                    )

                    # Load and convert to RGB if needed (for JPEG compatibility)
                    gen_img = Image.open(generated_images[0])
                    if gen_img.mode == 'RGBA':
                        gen_img = gen_img.convert('RGB')
                    gen_img.save(gen_image_dest)

                    result_entry["generated_image"] = f"./generated_images/{gen_image_filename}"

                self.simple_results.append(result_entry)

                # Save results to JSON file after each sample
                results_file = os.path.join(self.simple_output_dir, "results.json")
                with open(results_file, "w", encoding="utf-8") as f:
                    json.dump(self.simple_results, f, ensure_ascii=False, indent=2)

            # Return only final answer text
            res.append(final_answer)
            pbar.update(1)

        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """Not supported for generation models"""
        raise NotImplementedError(
            "OneCATVisualCoT is a generation model and does not support loglikelihood"
        )

    def generate_until_multi_round(self, requests) -> List[str]:
        """Multi-round dialogue not yet implemented"""
        raise NotImplementedError(
            "Multi-round dialogue not yet implemented for OneCATVisualCoT"
        )
