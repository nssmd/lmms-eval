import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from accelerate import Accelerator
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

# Add models directory to Python path
wd = Path(__file__).parent.parent.parent.parent.parent.resolve()
models_path = os.path.join(str(wd), "models")
if os.path.exists(models_path):
    sys.path.insert(0, models_path)
    eval_logger.info(f"Added models path to sys.path: {models_path}")
else:
    eval_logger.warning(
        f"Models directory not found at {models_path}. "
        f"Please ensure model files are available."
    )


@register_model("x_omni")
class XOmni(lmms):
    """
    X-Omni Multimodal Model
    Supports both image understanding and text-to-image generation

    Modes:
        - "understanding": Visual understanding (image + text -> text)
        - "generation": Image generation (text -> image)

    Example usage for understanding:
    accelerate launch -m lmms_eval \
        --model x_omni \
        --model_args pretrained=/path/to/X-Omni-En,mode=understanding,flux_pipe_path=/path/to/flux \
        --tasks mmbench \
        --batch_size 1 \
        --output_path ./logs/

    Example usage for generation:
    accelerate launch -m lmms_eval \
        --model x_omni \
        --model_args pretrained=/path/to/X-Omni-En,mode=generation,flux_pipe_path=/path/to/flux \
        --tasks ueval \
        --batch_size 1 \
        --output_path ./logs/
    """

    def __init__(
        self,
        pretrained: str,
        mode: str = "understanding",
        flux_pipe_path: Optional[str] = None,
        output_image_dir: Optional[str] = None,
        max_new_tokens: int = 512,
        do_sample: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.9,
        seed: int = 0,
        num_inference_steps: int = 28,
        guidance_scale: float = 3.5,
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
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale

        # Import X-Omni dependencies
        try:
            from transformers import AutoTokenizer

            from configuration_xomni import XOmniConfig
            from modeling_xomni import XOmniForCausalLM

            self.XOmniConfig = XOmniConfig
            self.XOmniForCausalLM = XOmniForCausalLM
            self.AutoTokenizer = AutoTokenizer

        except Exception as e:
            raise ImportError(
                f"Failed to import X-Omni dependencies. "
                f"Please ensure:\n"
                f"  1. Model files are in the models directory\n"
                f"  2. Required packages are installed\n"
                f"Error: {e}"
            )

        self.pretrained = pretrained
        self.flux_pipe_path = flux_pipe_path
        self.continual_mode = continual_mode

        # Validate flux_pipe_path for generation mode
        if mode == "generation" and flux_pipe_path is None:
            raise ValueError(
                "flux_pipe_path is required for generation mode. "
                "Please provide the path to the Flux pipeline."
            )

        # Setup output directory
        if response_persistent_folder is None:
            self.response_persistent_folder = "./logs/x_omni_persistent_folder"
        else:
            self.response_persistent_folder = response_persistent_folder

        if output_image_dir is None:
            self.output_image_dir = os.path.join(
                self.response_persistent_folder, "x_omni_generated_images"
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
                self.response_persistent_folder, "x_omni_response.json"
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
        eval_logger.info(f"Loading X-Omni model from {pretrained}")
        self._load_model()

        eval_logger.info("X-Omni model initialized successfully")

    def _load_model(self):
        """Load X-Omni model components"""
        model_path = self.pretrained

        # Load tokenizer
        eval_logger.info("Loading tokenizer...")
        self._tokenizer = self.AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )

        # Load model
        eval_logger.info("Loading model...")
        self._model = self.XOmniForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        # Initialize vision components
        eval_logger.info("Initializing vision components...")

        if self.mode == "understanding":
            # Understanding mode only needs encoder (no FLUX required)
            self._model.init_vision(flux_pipe_path=None)
            eval_logger.info("Vision encoder initialized for understanding mode")
        elif self.mode == "generation":
            # Generation mode needs both encoder and decoder (FLUX required)
            if self.flux_pipe_path is None:
                raise ValueError(
                    "flux_pipe_path is required for generation mode. "
                    "Please provide it via --model_args flux_pipe_path=/path/to/FLUX.1-dev"
                )
            self._model.init_vision(self.flux_pipe_path)
            eval_logger.info("Vision encoder and decoder initialized for generation mode")

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
    def tokenizer(self):
        return self._tokenizer

    def set_seed(self, seed: int):
        """Set random seeds for reproducibility"""
        if seed > 0:
            import random

            import numpy as np

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def understand_image(self, prompt: str, image: Image.Image, doc_id: str) -> str:
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

        # Set generation mode to text
        self.model.set_generation_mode("text")

        # Tokenize image first
        image_str = self.model.tokenize_image(image)

        # Create chat message with proper format
        message = [{'role': 'user', 'content': image_str + '\n' + prompt}]

        # Apply chat template
        input_ids = self.tokenizer.apply_chat_template(
            message,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        input_ids = input_ids.to(self.model.device)

        # Create attention mask
        attention_mask = torch.ones_like(input_ids)

        # Get proper EOS token
        eos_token_id = self.tokenizer.encode('<|im_end|>')[0]

        # Generate response
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                temperature=self.temperature if self.do_sample else 0.0,
                top_p=self.top_p if self.do_sample else None,
                eos_token_id=eos_token_id,
                pad_token_id=0,
                use_cache=True,
            )

        # Decode output using mmdecode
        texts, _ = self.model.mmdecode(self.tokenizer, output_ids[:, input_ids.shape[1]:-1])
        output_text = texts[0] if texts else ""

        return output_text

    def generate_image(
        self, prompt: str, doc_id: str, task: str
    ) -> Tuple[str, List[str]]:
        """
        Generate image from text prompt

        Args:
            prompt: Input text prompt
            doc_id: Document ID for file naming
            task: Task name for file naming

        Returns:
            Tuple of (generated_text, list_of_image_paths)
        """
        self.set_seed(self.seed)

        # Set generation mode to image
        self.model.set_generation_mode("image")

        # Encode input
        input_ids = self.model.mmencode(
            self.tokenizer, texts=[prompt], return_tensors="pt"
        )
        input_ids = input_ids.to(self.model.device)

        # Generate image tokens
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                temperature=self.temperature if self.do_sample else 1.0,
                top_p=self.top_p if self.do_sample else 1.0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode output (text and images)
        texts, images = self.model.mmdecode(self.tokenizer, output_ids)

        # Get generated text
        output_text = texts[-1] if texts else ""

        # Save generated images
        output_images = []
        for idx, image in enumerate(images):
            safe_filename = f"{task}_{doc_id}_{idx}.png"
            image_path = os.path.join(self.output_image_dir, safe_filename)
            image.save(image_path)
            output_images.append(image_path)
            eval_logger.info(f"Saved image: {image_path}")

        return output_text, output_images

    def format_output(self, text: str, images: List[str]) -> str:
        """Format output as JSON string"""
        output_dict = {"text": text, "images": images}
        result = json.dumps(output_dict, ensure_ascii=False)
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
            total=len(requests), disable=(self.rank != 0), desc="X-Omni Generating"
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

            # Debug: Log the first few prompts to check input
            if len(res) < 3:
                eval_logger.info(f"[DEBUG] Doc {doc_id} prompt: {prompt[:200]}...")

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
                output_text, output_images = self.generate_image(
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
            "X-Omni is a generation model and does not support loglikelihood"
        )

    def generate_until_multi_round(self, requests) -> List[str]:
        """Multi-round dialogue generation"""
        raise NotImplementedError("TODO: Implement multi-round dialogue generation")
