import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from accelerate import Accelerator
from loguru import logger as eval_logger
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

# Add mammothmoda repository to Python path
# Expected: lmms-eval/../mammothmoda/ directory (sibling to lmms-eval)
wd = Path(__file__).parent.parent.parent.parent.parent.resolve()
mammothmoda_path = os.path.join(str(wd), "mammothmoda")
if os.path.exists(mammothmoda_path):
    sys.path.append(mammothmoda_path)
    eval_logger.info(f"Added mammothmoda path to sys.path: {mammothmoda_path}")
else:
    eval_logger.warning(
        f"mammothmoda repository not found at {mammothmoda_path}. "
        f"Please ensure the mammothmoda directory exists at {wd}"
    )


@register_model("mammothmoda")
class MammothModa(lmms):
    """
    Mammothmoda Multimodal Model
    Supports both image understanding and text-to-image generation

    Modes:
        - "understanding": Visual understanding (image + text -> text)
        - "generation": Image generation (text -> image)

    Example usage for understanding:
    accelerate launch -m lmms_eval \
        --model mammothmoda \
        --model_args pretrained=/path/to/MammothModa2-Preview,mode=understanding \
        --tasks mmbench \
        --batch_size 1 \
        --output_path ./logs/

    Example usage for generation:
    accelerate launch -m lmms_eval \
        --model mammothmoda \
        --model_args pretrained=/path/to/MammothModa2-Preview,mode=generation \
        --tasks ueval \
        --batch_size 1 \
        --output_path ./logs/
    """

    def __init__(
        self,
        pretrained: str,
        mode: str = "generation",
        attn_implementation: str = "flash_attention_2",
        torch_dtype: str = "bfloat16",
        output_image_dir: Optional[str] = None,
        num_images_per_prompt: int = 1,
        cfg_scale: float = 7.0,
        text_guidance_scale: float = 9.0,
        num_inference_steps: int = 50,
        height: int = 1024,
        width: int = 1024,
        ar_height: int = 32,
        ar_width: int = 32,
        max_new_tokens: int = 512,
        do_sample: bool = False,
        temperature: float = 0.3,
        seed: int = 0,
        continual_mode: bool = True,
        response_persistent_folder: Optional[str] = None,
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

        # Import mammothmoda dependencies
        try:
            from mammothmoda2.model import DEFAULT_NEGATIVE_PROMPT, Mammothmoda2Model
            from mammothmoda2.utils import decode_diffusion_image
            from qwen_vl_utils import process_vision_info
            from transformers import AutoProcessor

            self.Mammothmoda2Model = Mammothmoda2Model
            self.AutoProcessor = AutoProcessor
            self.process_vision_info = process_vision_info
            self.decode_diffusion_image = decode_diffusion_image
            self.DEFAULT_NEGATIVE_PROMPT = DEFAULT_NEGATIVE_PROMPT

        except Exception as e:
            raise ImportError(
                f"Failed to import mammothmoda dependencies. "
                f"Please ensure:\n"
                f"  1. mammothmoda repository is available at {mammothmoda_path}\n"
                f"  2. Model weights are downloaded\n"
                f"Error: {e}"
            )

        self.pretrained = pretrained
        self.attn_implementation = attn_implementation
        self.torch_dtype_str = torch_dtype
        self.continual_mode = continual_mode

        # Convert torch_dtype string to actual dtype
        if torch_dtype == "bfloat16":
            self.torch_dtype = torch.bfloat16
        elif torch_dtype == "float16":
            self.torch_dtype = torch.float16
        elif torch_dtype == "float32":
            self.torch_dtype = torch.float32
        else:
            eval_logger.warning(f"Unknown torch_dtype {torch_dtype}, using bfloat16")
            self.torch_dtype = torch.bfloat16

        # Generation hyperparameters
        self.num_images_per_prompt = num_images_per_prompt
        self.cfg_scale = cfg_scale
        self.text_guidance_scale = text_guidance_scale
        self.num_inference_steps = num_inference_steps
        self.height = height
        self.width = width
        self.ar_height = ar_height
        self.ar_width = ar_width
        self.do_sample = do_sample
        self.temperature = temperature
        self.seed = seed

        # Setup output directory
        if response_persistent_folder is None:
            self.response_persistent_folder = "./logs/mammothmoda_persistent_folder"
        else:
            self.response_persistent_folder = response_persistent_folder

        if output_image_dir is None:
            self.output_image_dir = os.path.join(
                self.response_persistent_folder, "mammothmoda_generated_images"
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
                self.response_persistent_folder, "mammothmoda_response.json"
            )

            if os.path.exists(self.response_persistent_file):
                with open(self.response_persistent_file, "r") as f:
                    self.response_cache = json.load(f)
                self.cache_mode = "resume"
                eval_logger.info(f"Loaded cache: {len(self.response_cache)} records")

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
        eval_logger.info(f"Loading Mammothmoda model from {pretrained}")
        self._load_model()

        eval_logger.info("Mammothmoda model initialized successfully")

    def _load_model(self):
        """Load Mammothmoda model components"""
        model_path = self.pretrained

        # Load model based on mode
        if self.mode == "generation":
            # Load model with text-to-image generation support
            self._model = self.Mammothmoda2Model.from_pretrained(
                model_path,
                attn_implementation=self.attn_implementation,
                torch_dtype=self.torch_dtype_str,
                t2i_generate=True,
            ).to(self.accelerator.device)

            # Load processor with generation support
            self._processor = self.AutoProcessor.from_pretrained(
                model_path,
                t2i_generate=True,
                ar_height=self.ar_height,
                ar_width=self.ar_width,
            )
        else:
            # Load model for understanding mode
            self._model = self.Mammothmoda2Model.from_pretrained(
                model_path,
                attn_implementation=self.attn_implementation,
                torch_dtype=self.torch_dtype_str,
                t2i_generate=False,
            ).to(self.accelerator.device)

            # Load processor without generation support
            self._processor = self.AutoProcessor.from_pretrained(model_path)

        self._model.eval()

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
    def processor(self):
        return self._processor

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
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def understand_image(self, prompt: str, image, doc_id: str) -> str:
        """
        Understand image and answer question

        Args:
            prompt: Input text prompt/question
            image: PIL Image to understand
            doc_id: Document ID for logging

        Returns:
            Generated text answer
        """
        self.set_seed(self.seed)

        # Prepare messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Apply chat template
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = self.process_vision_info(messages)

        # Process inputs
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.accelerator.device)

        # Remove token_type_ids if present (not supported by model)
        inputs.pop("token_type_ids", None)

        # Generate response
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=self.torch_dtype
        ):
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                temperature=self.temperature,
            )

        # Decode output
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return output_text

    def generate_text_and_image(
        self, prompt: str, doc_id: str, task: str, image=None
    ) -> Tuple[str, List[str]]:
        """
        Generate text and image from prompt

        Args:
            prompt: Input text prompt
            doc_id: Document ID for file naming
            task: Task name for file naming
            image: Optional input image (not used in current implementation)

        Returns:
            Tuple of (generated_text, list_of_image_paths)
        """
        self.set_seed(self.seed)

        # Prepare messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Apply chat template
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = self.process_vision_info(messages)

        # Process inputs
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            num_images_per_prompt=self.num_images_per_prompt,
            cfg_scale=self.cfg_scale,
            negative_prompt=self.DEFAULT_NEGATIVE_PROMPT,
            padding=True,
            padding_side="left",
            return_tensors="pt",
            return_token_type_ids=False,
        ).to(self.accelerator.device)

        # Generate
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=self.torch_dtype
        ):
            generated_ids, attention_mask = self.model.generate(**inputs)

            # Decode diffusion images
            diff_return_info = self.decode_diffusion_image(
                input_ids=inputs.input_ids,
                generated_ids=generated_ids,
                attention_mask=attention_mask,
                negative_ids=inputs.get("negative_ids", None),
                negative_mask=inputs.get("negative_mask", None),
                model=self.model,
                tokenizer=self.processor.tokenizer,
                output_dir=self.output_image_dir,
                num_images_per_prompt=self.num_images_per_prompt,
                text_guidance_scale=self.text_guidance_scale,
                vae_scale_factor=16,
                cfg_range=(0.0, 1.0),
                num_inference_steps=self.num_inference_steps,
                height=self.height,
                width=self.width,
            )

        # Extract generated images
        # diff_return_info is a list of dicts with keys: "save_path", "decoded_img"
        output_images = []
        if isinstance(diff_return_info, list):
            for idx, img_info in enumerate(diff_return_info):
                if "decoded_img" in img_info:
                    img = img_info["decoded_img"]
                    safe_filename = f"{task}_{doc_id}_{idx}.png"
                    image_path = os.path.join(self.output_image_dir, safe_filename)
                    img.save(image_path)
                    output_images.append(image_path)
                    eval_logger.info(f"Saved image: {image_path}")

        # No text output for generation mode
        output_text = ""

        return output_text, output_images

    def format_output(self, text: str, images: List[str]) -> str:
        """Format output as JSON string"""
        eval_logger.debug(
            f"[FORMAT OUTPUT] Input: text type={type(text).__name__}, "
            f"text value={repr(text)}, images count={len(images) if images else 0}"
        )
        output_dict = {"text": text, "images": images}
        result = json.dumps(output_dict, ensure_ascii=False)
        eval_logger.debug(
            f"[FORMAT OUTPUT] Output JSON: {result[:200]}..."
            if len(result) > 200
            else f"[FORMAT OUTPUT] Output JSON: {result}"
        )
        return result

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
            desc="Mammothmoda Generating",
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
                output_text = self.understand_image(prompt, image, str(doc_id))
                formatted_output = output_text

            else:
                # Image generation mode
                output_text, output_images = self.generate_text_and_image(
                    prompt, str(doc_id), task
                )
                formatted_output = self.format_output(output_text, output_images)

            res.append(formatted_output)

            # Update cache
            if self.continual_mode:
                self.response_cache[doc_uuid] = formatted_output
                with open(self.response_persistent_file, "w") as f:
                    json.dump(self.response_cache, f, ensure_ascii=False, indent=2)

            pbar.update(1)

        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """Not supported for generation models"""
        raise NotImplementedError(
            "Mammothmoda is a generation model and does not support loglikelihood"
        )

    def generate_until_multi_round(self, requests) -> List[str]:
        """Multi-round dialogue generation"""
        raise NotImplementedError("TODO: Implement multi-round dialogue generation")
