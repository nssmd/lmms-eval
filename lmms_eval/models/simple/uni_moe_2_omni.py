"""
Uni-MoE-2.0-Omni Multimodal Model Integration

Uni-MoE 2.0 is a fully open-source omnimodal model powered by Omnimodality 3D RoPE
and Dynamic-Capacity Mixture-of-Experts architecture. It supports:
- All-modality understanding (image, video, audio, text)
- Speech generation (TTS)
- Image generation, editing, and low-level image processing

Paper: https://arxiv.org/abs/2511.12609
Model: https://huggingface.co/HIT-TMG/Uni-MoE-2.0-Omni
GitHub: https://github.com/HITsz-TMG/Uni-MoE

Usage for understanding:
    python -m lmms_eval \
        --model uni_moe_2_omni \
        --model_args pretrained=HIT-TMG/Uni-MoE-2.0-Omni,mode=understanding \
        --tasks mmbench \
        --batch_size 1 \
        --output_path ./logs/

Usage for generation:
    python -m lmms_eval \
        --model uni_moe_2_omni \
        --model_args pretrained=HIT-TMG/Uni-MoE-2.0-Omni,mode=generation \
        --tasks geneval \
        --batch_size 1 \
        --output_path ./logs/
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

# Add Uni-MoE repository to Python path
# Expected: lmms-eval/Uni-MoE/Uni-MoE-2/ directory at project root
wd = Path(__file__).parent.parent.parent.parent.resolve()
uni_moe_path = os.path.join(str(wd), "Uni-MoE", "Uni-MoE-2")
if os.path.exists(uni_moe_path):
    sys.path.insert(0, uni_moe_path)
    eval_logger.info(f"Added Uni-MoE path to sys.path: {uni_moe_path}")
else:
    eval_logger.warning(
        f"Uni-MoE repository not found at {uni_moe_path}. "
        f"Please clone it: cd {wd} && git clone https://github.com/HITsz-TMG/Uni-MoE.git"
    )


@register_model("uni_moe_2_omni")
class UniMoE2Omni(lmms):
    """
    Uni-MoE-2.0-Omni: Scaling Language-Centric Omnimodal Large Model

    Supports:
    - Image/Video/Audio understanding (understanding mode)
    - Image generation (generation mode)

    Based on Qwen2.5-7B-Instruct with MoE architecture.
    """

    def __init__(
        self,
        pretrained: str = "HIT-TMG/Uni-MoE-2.0-Omni",
        mode: str = "understanding",
        device: str = "cuda",
        device_map: str = "auto",
        batch_size: int = 1,
        # Text generation parameters
        max_new_tokens: int = 4096,
        do_sample: bool = False,
        temperature: float = 0.0,
        num_beams: int = 1,
        # Image generation parameters (for generation mode)
        image_generation_steps: int = 50,
        image_guidance_scale: float = 7.5,
        # Model loading
        use_flash_attention_2: bool = True,
        torch_dtype: str = "bfloat16",
        trust_remote_code: bool = True,
        # Audio in video
        use_audio_in_video: bool = False,
        # Thinking mode (for Uni-MoE-2.0-Thinking)
        think_mode: bool = False,
        # Output
        output_image_dir: Optional[str] = None,
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

        self.pretrained = pretrained
        self.mode = mode
        self.device_str = device
        self.device_map = device_map
        self.batch_size_per_gpu = batch_size
        self.trust_remote_code = trust_remote_code

        # Text generation parameters
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.temperature = temperature
        self.num_beams = num_beams

        # Image generation parameters
        self.image_generation_steps = image_generation_steps
        self.image_guidance_scale = image_guidance_scale

        # Audio in video
        self.use_audio_in_video = use_audio_in_video

        # Thinking mode
        self.think_mode = think_mode

        # Flash attention
        self.use_flash_attention_2 = use_flash_attention_2

        # Set torch dtype
        if torch_dtype == "bfloat16":
            self.torch_dtype = torch.bfloat16
        elif torch_dtype == "float16":
            self.torch_dtype = torch.float16
        else:
            self.torch_dtype = torch.float32

        # Setup output directory
        if response_persistent_folder is None:
            self.response_persistent_folder = "./logs/uni_moe_2_omni_persistent_folder"
        else:
            self.response_persistent_folder = response_persistent_folder

        if output_image_dir is None:
            self.output_image_dir = os.path.join(
                self.response_persistent_folder, "uni_moe_2_omni_generated_images"
            )
        else:
            self.output_image_dir = output_image_dir

        if self.mode == "generation":
            os.makedirs(self.output_image_dir, exist_ok=True)
            eval_logger.info(f"Image output directory: {self.output_image_dir}")

        # Setup response cache for continual mode
        self.continual_mode = continual_mode
        self.response_cache = {}
        self.cache_mode = "start"

        if self.continual_mode:
            os.makedirs(self.response_persistent_folder, exist_ok=True)
            self.response_persistent_file = os.path.join(
                self.response_persistent_folder, "uni_moe_2_omni_response.json"
            )

            if os.path.exists(self.response_persistent_file):
                with open(self.response_persistent_file, "r") as f:
                    self.response_cache = json.load(f)
                self.cache_mode = "resume"
                eval_logger.info(f"Loaded cache: {len(self.response_cache)} records")

        # Setup accelerator for distributed inference
        from accelerate import Accelerator

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
        eval_logger.info(f"Loading Uni-MoE-2.0-Omni model from {pretrained}")
        self._load_model()
        eval_logger.info("Uni-MoE-2.0-Omni initialized successfully")

    def _load_model(self):
        """Load Uni-MoE-2.0-Omni model and processor"""
        try:
            from uni_moe.model.modeling_out import GrinQwen2VLOutForConditionalGeneration
            from uni_moe.model.processing_qwen2_vl import Qwen2VLProcessor
            from uni_moe.qwen_vl_utils import process_mm_info

            self.process_mm_info = process_mm_info
        except ImportError:
            eval_logger.warning(
                "Failed to import from uni_moe package. "
                "Trying to import from transformers with trust_remote_code=True"
            )
            from transformers import AutoModelForCausalLM, AutoProcessor

            # Fallback to transformers auto classes
            self.process_mm_info = None

        # Setup attention implementation
        attn_implementation = (
            "flash_attention_2" if self.use_flash_attention_2 else "eager"
        )
        if self.use_flash_attention_2:
            try:
                import flash_attn  # noqa: F401

                eval_logger.info("Using Flash Attention 2")
            except ImportError:
                eval_logger.warning(
                    "flash_attn not installed, falling back to eager attention"
                )
                attn_implementation = "eager"

        # Determine device
        if self.device_map == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            device = self.device_str if torch.cuda.is_available() else "cpu"
        self.device = device

        # Load processor
        try:
            from uni_moe.model.processing_qwen2_vl import Qwen2VLProcessor

            self.processor = Qwen2VLProcessor.from_pretrained(
                self.pretrained,
                trust_remote_code=self.trust_remote_code,
            )
        except ImportError:
            from transformers import AutoProcessor

            self.processor = AutoProcessor.from_pretrained(
                self.pretrained,
                trust_remote_code=self.trust_remote_code,
            )

        # Load model
        try:
            from uni_moe.model.modeling_out import GrinQwen2VLOutForConditionalGeneration

            if self.device_map == "auto":
                self.model = GrinQwen2VLOutForConditionalGeneration.from_pretrained(
                    self.pretrained,
                    device_map="auto",
                    torch_dtype=self.torch_dtype,
                    attn_implementation=attn_implementation,
                    trust_remote_code=self.trust_remote_code,
                ).eval()
            else:
                self.model = (
                    GrinQwen2VLOutForConditionalGeneration.from_pretrained(
                        self.pretrained,
                        torch_dtype=self.torch_dtype,
                        attn_implementation=attn_implementation,
                        trust_remote_code=self.trust_remote_code,
                    )
                    .to(device)
                    .eval()
                )
        except ImportError:
            from transformers import AutoModelForCausalLM

            if self.device_map == "auto":
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.pretrained,
                    device_map="auto",
                    torch_dtype=self.torch_dtype,
                    attn_implementation=attn_implementation,
                    trust_remote_code=self.trust_remote_code,
                ).eval()
            else:
                self.model = (
                    AutoModelForCausalLM.from_pretrained(
                        self.pretrained,
                        torch_dtype=self.torch_dtype,
                        attn_implementation=attn_implementation,
                        trust_remote_code=self.trust_remote_code,
                    )
                    .to(device)
                    .eval()
                )

        # Set processor data_args
        if hasattr(self.processor, "data_args"):
            self.processor.data_args = self.model.config

        eval_logger.info(f"Model loaded on {device}")

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def tokenizer(self):
        return self.processor.tokenizer

    def flatten(self, input_list):
        """Flatten nested lists"""
        output = []
        for item in input_list:
            if isinstance(item, list):
                output.extend(self.flatten(item))
            else:
                output.append(item)
        return output

    def _prepare_inputs(self, messages: List[dict]) -> dict:
        """
        Prepare inputs from messages format

        Args:
            messages: List of message dicts with role and content

        Returns:
            Processed inputs dict ready for model
        """
        # Apply chat template
        texts = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Replace placeholder tokens with actual tokens
        texts = (
            texts.replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>")
            .replace("<audio>", "<|audio_start|><|audio_pad|><|audio_end|>")
            .replace("<video>", "<|vision_start|><|video_pad|><|vision_end|>")
        )

        # Process multimodal info
        if self.process_mm_info is not None:
            image_inputs, video_inputs, audio_inputs = self.process_mm_info(messages)
        else:
            # Fallback: extract from messages manually
            image_inputs = []
            video_inputs = []
            audio_inputs = []
            for msg in messages:
                if isinstance(msg.get("content"), list):
                    for item in msg["content"]:
                        if item.get("type") == "image":
                            img = item.get("image")
                            if isinstance(img, str):
                                image_inputs.append(Image.open(img).convert("RGB"))
                            elif isinstance(img, Image.Image):
                                image_inputs.append(img.convert("RGB"))
                        elif item.get("type") == "video":
                            video_inputs.append(item.get("video"))
                        elif item.get("type") == "audio":
                            audio_inputs.append(item.get("audio"))

        # Process with processor
        inputs = self.processor(
            text=texts,
            images=image_inputs if image_inputs else None,
            videos=video_inputs if video_inputs else None,
            audios=audio_inputs if audio_inputs else None,
            padding=True,
            return_tensors="pt",
        )

        # Add batch dimension if needed
        if "input_ids" in inputs and inputs["input_ids"].dim() == 1:
            inputs["input_ids"] = inputs["input_ids"].unsqueeze(0)

        return inputs

    def _build_messages_from_visuals_and_text(
        self,
        visuals: List,
        text: str,
    ) -> List[dict]:
        """
        Build messages format from visuals and text

        Args:
            visuals: List of PIL Images or paths
            text: Text prompt/question

        Returns:
            Messages in the format expected by the model
        """
        content = []

        # Add images
        for i, visual in enumerate(visuals):
            if isinstance(visual, Image.Image):
                content.append({"type": "image", "image": visual})
            elif isinstance(visual, str):
                # Could be path to image or video
                if visual.endswith((".mp4", ".avi", ".mov", ".mkv")):
                    content.append({"type": "video", "video": visual})
                else:
                    content.append({"type": "image", "image": visual})

        # Build text with image placeholders
        if visuals:
            # Add image placeholders to text
            image_placeholders = "".join(
                ["<image>\n" for _ in range(len(visuals)) if not isinstance(visuals[0], str) or not visuals[0].endswith((".mp4", ".avi", ".mov", ".mkv"))]
            )
            text_with_placeholders = image_placeholders + text
        else:
            text_with_placeholders = text

        content.append({"type": "text", "text": text_with_placeholders})

        messages = [{"role": "user", "content": content}]
        return messages

    def understand_image(
        self, prompt: str, visuals: List, doc_id: str
    ) -> str:
        """
        Understand image(s) and answer question

        Args:
            prompt: Input text prompt/question
            visuals: List of PIL Images or paths
            doc_id: Document ID for logging

        Returns:
            Generated text answer
        """
        # Build messages
        messages = self._build_messages_from_visuals_and_text(visuals, prompt)

        # Prepare inputs
        inputs = self._prepare_inputs(messages)
        inputs = inputs.to(device=self.model.device)

        # Convert to bfloat16 for specific tensors
        for k, v in inputs.items():
            if k in ["pixel_values", "pixel_values_videos", "audio_features"]:
                if v is not None:
                    inputs[k] = v.to(dtype=self.torch_dtype)

        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                use_cache=True,
                pad_token_id=self.processor.tokenizer.eos_token_id,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                temperature=self.temperature if self.do_sample else None,
                num_beams=self.num_beams,
            )

        # Decode output
        generated_ids = output_ids[:, inputs["input_ids"].shape[-1]:]
        output_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        return output_text

    def generate_image(
        self, prompt: str, doc_id: str, task: str
    ) -> Tuple[str, List[str]]:
        """
        Generate image from text prompt

        Args:
            prompt: Text prompt for image generation
            doc_id: Document ID for file naming
            task: Task name for file naming

        Returns:
            Tuple of (generated_text, list_of_image_paths)
        """
        # Build messages for image generation
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Prepare inputs
        inputs = self._prepare_inputs(messages)
        inputs = inputs.to(device=self.model.device)

        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                use_cache=True,
                pad_token_id=self.processor.tokenizer.eos_token_id,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=1.0,
            )

        # Decode output
        generated_ids = output_ids[:, inputs["input_ids"].shape[-1]:]
        output_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        # TODO: Implement actual image generation using diffusion model
        # For now, return empty image list
        output_images = []

        return output_text, output_images

    def format_output(self, text: str, images: List[str]) -> str:
        """Format output as JSON string"""
        output_dict = {"text": text, "images": images}
        return json.dumps(output_dict, ensure_ascii=False)

    def generate_until(self, requests: List[Instance]) -> List[str]:
        """Main inference method"""
        res = []
        pbar = tqdm(
            total=len(requests),
            disable=(self.rank != 0),
            desc="Uni-MoE-2.0-Omni Generating",
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
                visuals = []
                if doc_to_visual is not None:
                    try:
                        doc = self.task_dict[task][split][doc_id]
                        visuals = doc_to_visual(doc)
                        if visuals:
                            visuals = self.flatten([visuals])
                    except Exception as e:
                        eval_logger.warning(
                            f"Failed to get visuals for doc_id={doc_id}: {e}"
                        )

                output_text = self.understand_image(prompt, visuals, str(doc_id))
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
            "Uni-MoE-2.0-Omni is a generation model and does not support loglikelihood"
        )

    def generate_until_multi_round(self, requests: List[Instance]) -> List[str]:
        """Multi-round dialogue generation"""
        raise NotImplementedError(
            "Multi-round dialogue not yet implemented for Uni-MoE-2.0-Omni"
        )
