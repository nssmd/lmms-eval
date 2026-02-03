"""
OmniGen2 Model - Unified Multimodal Generation

OmniGen2 is a unified multimodal model that combines:
- 3B Vision-Language Model (Qwen2.5-VL) for understanding
- 4B Diffusion Transformer for image generation

Paper: https://arxiv.org/abs/2506.18871
GitHub: https://github.com/VectorSpaceLab/OmniGen2
HuggingFace: https://huggingface.co/OmniGen2/OmniGen2

Modes:
    - "understanding": Visual understanding (image + text -> text)
    - "generation": Image generation (text -> image) or (image + text -> image)

Example usage for understanding:
    python -m lmms_eval --model omnigen2 \
        --model_args pretrained=OmniGen2/OmniGen2,mode=understanding \
        --tasks mmbench --batch_size 1 --device cuda:0

Example usage for generation:
    python -m lmms_eval --model omnigen2 \
        --model_args pretrained=OmniGen2/OmniGen2,mode=generation \
        --tasks geneval --batch_size 1 --device cuda:0
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

# Add OmniGen2 repository to Python path
# Try multiple possible locations
wd = Path(__file__).parent.parent.parent.parent.resolve()
possible_paths = [
    os.path.join(str(wd), "OmniGen2"),  # lmms-eval/OmniGen2
    os.path.join(str(wd.parent), "OmniGen2"),  # parent/OmniGen2
    os.path.expanduser("~/data/zwb/OmniGen2"),  # ~/data/zwb/OmniGen2
    "/home/aiscuser/data/zwb/OmniGen2",  # Absolute path
]

omnigen2_path = None
for path in possible_paths:
    if os.path.exists(path):
        omnigen2_path = path
        break

if omnigen2_path:
    sys.path.insert(0, omnigen2_path)
    eval_logger.info(f"Added OmniGen2 path to sys.path: {omnigen2_path}")
else:
    eval_logger.warning(
        f"OmniGen2 repository not found. Tried: {possible_paths}. "
        f"Please clone it: git clone https://github.com/VectorSpaceLab/OmniGen2.git"
    )


@register_model("omnigen2")
class OmniGen2(lmms):
    """
    OmniGen2 Model - Unified Multimodal Generation

    Architecture:
        - 3B VLM: Qwen2.5-VL for text/image understanding
        - 4B Diffusion Transformer for image generation
        - VAE for image encoding/decoding
    """

    def __init__(
        self,
        pretrained: str = "OmniGen2/OmniGen2",
        mode: str = "generation",
        device: str = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = torch.bfloat16,
        batch_size: int = 1,
        # Generation parameters
        num_inference_steps: int = 50,
        text_guidance_scale: float = 4.0,
        image_guidance_scale: float = 1.5,
        cfg_range: Tuple[float, float] = (0.0, 1.0),
        height: int = 1024,
        width: int = 1024,
        max_pixels: int = 1024 * 1024,
        max_input_image_side_length: int = 1024,
        negative_prompt: str = "blurry, low quality, text, watermark",
        # Understanding parameters
        max_new_tokens: int = 512,
        # Output settings
        output_image_dir: Optional[str] = None,
        response_persistent_folder: Optional[str] = None,
        continual_mode: bool = True,
        # Memory optimization
        enable_model_cpu_offload: bool = False,
        enable_teacache: bool = False,
        teacache_thresh: float = 0.05,
        seed: int = 42,
        **kwargs,
    ) -> None:
        super().__init__()

        # Validate mode
        if mode not in ["understanding", "generation"]:
            raise ValueError(
                f"mode must be 'understanding' or 'generation', got '{mode}'"
            )

        self.pretrained = pretrained
        self.mode = mode
        self.batch_size_per_gpu = int(batch_size)
        self.seed = seed

        # Generation parameters
        self.num_inference_steps = num_inference_steps
        self.text_guidance_scale = text_guidance_scale
        self.image_guidance_scale = image_guidance_scale
        self.cfg_range = cfg_range
        self.height = height
        self.width = width
        self.max_pixels = max_pixels
        self.max_input_image_side_length = max_input_image_side_length
        self.negative_prompt = negative_prompt

        # Understanding parameters
        self.max_new_tokens = max_new_tokens

        # Memory optimization
        self.enable_model_cpu_offload = enable_model_cpu_offload
        self.enable_teacache = enable_teacache
        self.teacache_thresh = teacache_thresh

        # Set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Determine device
        if torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            if world_size > 1:
                self._device = torch.device(f"cuda:{local_rank}")
            else:
                self._device = (
                    device if isinstance(device, torch.device) else torch.device(device)
                )
        else:
            self._device = torch.device("cpu")

        # Determine dtype
        if dtype == "auto" or dtype is None:
            self._dtype = torch.bfloat16
        elif isinstance(dtype, str):
            dtype_map = {
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }
            self._dtype = dtype_map.get(dtype, torch.bfloat16)
        else:
            self._dtype = dtype

        # Setup output directories
        if response_persistent_folder is None:
            self.response_persistent_folder = "./logs/omnigen2_persistent_folder"
        else:
            self.response_persistent_folder = response_persistent_folder

        if output_image_dir is None:
            self.output_image_dir = os.path.join(
                self.response_persistent_folder, "omnigen2_generated_images"
            )
        else:
            self.output_image_dir = output_image_dir

        os.makedirs(self.output_image_dir, exist_ok=True)
        eval_logger.info(f"Image output directory: {self.output_image_dir}")

        # Setup response cache for continual mode
        self.response_cache = {}
        self.cache_mode = "start"
        self.continual_mode = continual_mode

        if self.continual_mode:
            os.makedirs(self.response_persistent_folder, exist_ok=True)
            self.response_persistent_file = os.path.join(
                self.response_persistent_folder, "omnigen2_response.json"
            )
            if os.path.exists(self.response_persistent_file):
                with open(self.response_persistent_file, "r") as f:
                    self.response_cache = json.load(f)
                self.cache_mode = "resume"
                eval_logger.info(f"Loaded cache: {len(self.response_cache)} records")

        # Setup distributed training
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        if world_size > 1:
            from accelerate import Accelerator

            accelerator = Accelerator()
            self.accelerator = accelerator
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
            if self.continual_mode:
                eval_logger.warning(
                    "Continual mode is not supported for distributed inference. "
                    "Automatically disabling continual_mode."
                )
                self.continual_mode = False
        else:
            self._rank = 0
            self._world_size = 1

        # Load model
        eval_logger.info(f"Loading OmniGen2 model from {pretrained}")
        self._load_model()

        eval_logger.info("OmniGen2 model initialized successfully")

    def _load_model(self):
        """Load OmniGen2 model pipeline"""
        try:
            from omnigen2.pipelines.omnigen2.pipeline_omnigen2_chat import (
                OmniGen2ChatPipeline,
            )
            from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
        except ImportError as e:
            raise ImportError(
                f"Failed to import OmniGen2. "
                f"Please ensure OmniGen2 repository is cloned at lmms-eval root: "
                f"git clone https://github.com/VectorSpaceLab/OmniGen2.git\n"
                f"Error: {e}"
            )

        # Use ChatPipeline for understanding mode, regular Pipeline for generation
        if self.mode == "understanding":
            self._pipeline = OmniGen2ChatPipeline.from_pretrained(
                self.pretrained,
                torch_dtype=self._dtype,
                trust_remote_code=True,
            )
        else:
            self._pipeline = OmniGen2Pipeline.from_pretrained(
                self.pretrained,
                torch_dtype=self._dtype,
                trust_remote_code=True,
            )

        # Apply memory optimizations
        if self.enable_model_cpu_offload:
            self._pipeline.enable_model_cpu_offload()
            eval_logger.info("Enabled model CPU offload")
        else:
            self._pipeline = self._pipeline.to(self._device)

        # Ensure transformer has enable_teacache attribute (required by OmniGen2 pipeline)
        # The HuggingFace version may not have this attribute
        if hasattr(self._pipeline, "transformer"):
            if not hasattr(self._pipeline.transformer, "enable_teacache"):
                self._pipeline.transformer.enable_teacache = False

        # Enable TeaCache for faster inference (only if supported)
        if self.enable_teacache and hasattr(self._pipeline, "transformer"):
            if hasattr(self._pipeline.transformer, "teacache_rel_l1_thresh"):
                self._pipeline.transformer.enable_teacache = True
                self._pipeline.transformer.teacache_rel_l1_thresh = self.teacache_thresh
                eval_logger.info(
                    f"Enabled TeaCache with threshold {self.teacache_thresh}"
                )
            else:
                eval_logger.warning(
                    "TeaCache not fully supported by this transformer version, skipping"
                )

        self._tokenizer = self._pipeline.processor.tokenizer

        eval_logger.info(f"OmniGen2 pipeline loaded in {self.mode} mode")

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    @property
    def model(self):
        return self._pipeline

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    def flatten(self, input_list: List) -> List:
        """Flatten a nested list"""
        output = []
        for item in input_list:
            if isinstance(item, list):
                output.extend(self.flatten(item))
            else:
                output.append(item)
        return output

    def _extract_image(self, visual) -> Optional[Image.Image]:
        """Extract PIL Image from various formats"""
        try:
            if visual is None:
                return None
            elif isinstance(visual, Image.Image):
                return visual.convert("RGB")
            elif isinstance(visual, str):
                return Image.open(visual).convert("RGB")
            elif isinstance(visual, dict):
                if "bytes" in visual:
                    from io import BytesIO

                    return Image.open(BytesIO(visual["bytes"])).convert("RGB")
                elif "path" in visual:
                    return Image.open(visual["path"]).convert("RGB")
                elif "image" in visual:
                    return self._extract_image(visual["image"])
            return None
        except Exception as e:
            eval_logger.warning(f"Failed to extract image: {e}")
            return None

    def understand_image(
        self, prompt: str, images: Optional[List[Image.Image]], doc_id: str
    ) -> str:
        """
        Understand image and answer question using OmniGen2

        Args:
            prompt: Input text prompt/question
            images: List of PIL Images to understand (optional)
            doc_id: Document ID for logging

        Returns:
            Generated text answer
        """
        # Use ChatPipeline's generate_text method
        formatted_prompt = self._pipeline._apply_chat_template(prompt, images)
        output_texts = self._pipeline.generate_text(formatted_prompt, images)

        if output_texts:
            response = output_texts[0]
            # Clean up response
            response = response.replace("<|im_end|>", "").strip()
            return response
        return ""

    def generate_image(
        self,
        prompt: str,
        doc_id: str,
        task: str,
        images: Optional[List[Image.Image]] = None,
    ) -> Tuple[str, List[str]]:
        """
        Generate image from text prompt (optionally conditioned on input images)

        Args:
            prompt: Text prompt for generation
            doc_id: Document ID for file naming
            task: Task name for file naming
            images: Optional input images for conditioning

        Returns:
            Tuple of (prompt, list_of_image_paths)
        """
        if self.mode != "generation":
            raise RuntimeError(
                "generate_image requires mode='generation'. "
                f"Current mode is '{self.mode}'."
            )

        # Set seed for reproducibility
        generator = torch.Generator(device=self._device).manual_seed(self.seed)

        # Generate image
        with torch.no_grad():
            result = self._pipeline(
                prompt=prompt,
                input_images=images,
                height=self.height,
                width=self.width,
                max_pixels=self.max_pixels,
                max_input_image_side_length=self.max_input_image_side_length,
                num_inference_steps=self.num_inference_steps,
                text_guidance_scale=self.text_guidance_scale,
                image_guidance_scale=self.image_guidance_scale,
                cfg_range=self.cfg_range,
                negative_prompt=self.negative_prompt,
                generator=generator,
                return_dict=True,
            )

        # Save generated images
        output_images = []
        if result.images:
            for i, img in enumerate(result.images):
                if i == 0:
                    safe_filename = f"{task}_{doc_id}.png"
                else:
                    safe_filename = f"{task}_{doc_id}_{i}.png"
                image_path = os.path.join(self.output_image_dir, safe_filename)
                img.save(image_path)
                output_images.append(image_path)
                eval_logger.info(f"Saved generated image: {image_path}")

        return prompt, output_images

    def format_output(self, text: str, images: List[str]) -> str:
        """Format output as JSON string"""
        output_dict = {"text": text, "images": images}
        return json.dumps(output_dict, ensure_ascii=False)

    def generate_until(self, requests: List[Instance]) -> List[str]:
        """Main inference method"""
        res = []
        pbar = tqdm(
            total=len(requests), disable=(self.rank != 0), desc="OmniGen2 Generating"
        )

        def get_uuid(task, split, doc_id):
            return f"{task}___{split}___{doc_id}"

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        # Group requests
        re_ords = utils.Collator(
            [reg.args for reg in requests], _collate, grouping=True
        )
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)

        for chunk in chunks:
            (
                contexts,
                all_gen_kwargs,
                doc_to_visual,
                doc_id,
                task,
                split,
            ) = zip(*chunk)
            task = task[0]
            split = split[0]
            doc_id = doc_id[0]
            context = contexts[0]

            doc_uuid = get_uuid(task, split, doc_id)

            # Check cache
            if self.continual_mode and self.cache_mode == "resume":
                if doc_uuid in self.response_cache:
                    content = self.response_cache[doc_uuid]
                    if content:
                        res.append(content)
                        pbar.update(1)
                        continue

            # Get visuals
            visuals = [doc_to_visual[0](self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)

            # Extract images
            images = []
            for v in visuals:
                img = self._extract_image(v)
                if img is not None:
                    images.append(img)

            if self.mode == "understanding":
                # Image understanding mode
                output_text = self.understand_image(
                    context, images if images else None, str(doc_id)
                )
                formatted_output = output_text
            else:
                # Image generation mode
                output_text, output_images = self.generate_image(
                    context, str(doc_id), task, images if images else None
                )
                formatted_output = self.format_output(output_text, output_images)

            res.append(formatted_output)

            # Update cache
            if self.continual_mode:
                self.response_cache[doc_uuid] = formatted_output
                with open(self.response_persistent_file, "w") as f:
                    json.dump(self.response_cache, f, ensure_ascii=False, indent=2)

            pbar.update(1)

        # Reorder results
        res = re_ords.get_original(res)
        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """Not supported for generation models"""
        raise NotImplementedError(
            "OmniGen2 is a generation model and does not support loglikelihood"
        )

    def generate_until_multi_round(self, requests) -> List[str]:
        """Multi-round dialogue not yet implemented"""
        raise NotImplementedError(
            "Multi-round dialogue not yet implemented for OmniGen2"
        )
