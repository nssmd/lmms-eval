"""
AIA (Attention Interaction Alignment) Model Integration for lmms-eval.

Based on the paper: "Architecture Decoupling Is Not All You Need For Unified Multimodal Model"
AIA is built on top of Janus-Pro-7B with attention interaction alignment training.

Supports both image understanding and text-to-image generation modes.
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import PIL.Image
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

# Add AIA repository to Python path
wd = Path(__file__).parent.parent.parent.parent.resolve()
aia_path = os.path.join(str(wd), "AIA")
if os.path.exists(aia_path):
    sys.path.insert(0, aia_path)
    eval_logger.info(f"Added AIA path to sys.path: {aia_path}")
else:
    eval_logger.warning(
        f"AIA repository not found at {aia_path}. "
        f"Please clone it or set the correct path."
    )


@register_model("aia")
class AIA(lmms):
    """
    AIA (Attention Interaction Alignment) Multimodal Model.

    Supports both image understanding and text-to-image generation.

    Modes:
        - "understanding": Visual understanding (image + text -> text)
        - "generation": Image generation (text -> image)

    Example usage for understanding:
    accelerate launch -m lmms_eval \\
        --model aia \\
        --model_args pretrained=deepseek-ai/Janus-Pro-7B,ckpt_path=/path/to/aia.pt,mode=understanding \\
        --tasks mmbench \\
        --batch_size 1 \\
        --output_path ./logs/

    Example usage for generation:
    accelerate launch -m lmms_eval \\
        --model aia \\
        --model_args pretrained=deepseek-ai/Janus-Pro-7B,ckpt_path=/path/to/aia.pt,mode=generation \\
        --tasks geneval \\
        --batch_size 1 \\
        --output_path ./logs/
    """

    def __init__(
        self,
        pretrained: str = "deepseek-ai/Janus-Pro-7B",
        ckpt_path: Optional[str] = None,
        mode: str = "understanding",
        device: str = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "bfloat16",
        batch_size: int = 1,
        trust_remote_code: bool = True,
        use_cache: bool = True,
        max_new_tokens: int = 512,
        attn_implementation: Optional[str] = "eager",
        # Generation parameters
        output_image_dir: Optional[str] = None,
        cfg_weight: float = 5.0,
        temperature: float = 1.0,
        parallel_size: int = 1,
        image_token_num: int = 576,
        img_size: int = 384,
        patch_size: int = 16,
        # Continual mode for caching
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
        self.pretrained = pretrained
        self.ckpt_path = ckpt_path
        self.max_new_tokens = max_new_tokens
        self.use_cache = use_cache
        self.trust_remote_code = trust_remote_code
        self.attn_implementation = attn_implementation

        # Generation parameters
        self.cfg_weight = cfg_weight
        self.temperature = temperature
        self.parallel_size = parallel_size
        self.image_token_num = image_token_num
        self.img_size = img_size
        self.patch_size = patch_size
        self.continual_mode = continual_mode

        # Setup accelerator
        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
        else:
            self._device = device

        # Determine dtype
        if dtype in ("bfloat16", "bf16"):
            self._dtype = torch.bfloat16
        elif dtype in ("float16", "fp16"):
            self._dtype = torch.float16
        elif dtype in ("float32", "fp32"):
            self._dtype = torch.float32
        else:
            self._dtype = torch.bfloat16

        # Setup output directory for generation
        if response_persistent_folder is None:
            self.response_persistent_folder = "./logs/aia_persistent_folder"
        else:
            self.response_persistent_folder = response_persistent_folder

        if output_image_dir is None:
            self.output_image_dir = os.path.join(
                self.response_persistent_folder, "aia_generated_images"
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
                self.response_persistent_folder, "aia_response.json"
            )
            if os.path.exists(self.response_persistent_file):
                with open(self.response_persistent_file, "r") as f:
                    self.response_cache = json.load(f)
                self.cache_mode = "resume"
                eval_logger.info(f"Loaded cache: {len(self.response_cache)} records")

        # Load model
        eval_logger.info(f"Loading AIA model from {pretrained}")
        if ckpt_path:
            eval_logger.info(f"Loading AIA checkpoint from {ckpt_path}")
        self._load_model()

        self.batch_size_per_gpu = int(batch_size)
        assert self.batch_size_per_gpu == 1, "batch_size > 1 not supported for AIA"

        # Setup distributed training
        if accelerator.num_processes > 1:
            if self.continual_mode:
                eval_logger.warning(
                    "Continual mode is not supported for distributed inference. "
                    "Automatically disabling continual_mode."
                )
                self.continual_mode = False
            distributed_type_list = [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
                DistributedType.DEEPSPEED,
            ]
            assert accelerator.distributed_type in distributed_type_list, (
                "Unsupported distributed type. Only DDP, FSDP, and DeepSpeed supported"
            )
            self.accelerator = accelerator
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.accelerator = accelerator
            self._rank = 0
            self._world_size = 1

        eval_logger.info(f"AIA model initialized successfully in {mode} mode")

    def _load_model(self):
        """Load AIA model and processor."""
        try:
            from transformers import AutoModelForCausalLM

            from models import MultiModalityCausalLM, VLChatProcessor
        except ImportError as e:
            raise ImportError(
                f"Failed to import AIA dependencies. "
                f"Please ensure AIA repository is in the path.\n"
                f"Error: {e}"
            )

        # Load processor
        self._processor = VLChatProcessor.from_pretrained(
            self.pretrained, attn_implementation=self.attn_implementation
        )
        self._tokenizer = self._processor.tokenizer

        # Load base model
        self._model: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
            self.pretrained, trust_remote_code=self.trust_remote_code
        )

        # Load fine-tuned checkpoint if provided
        if self.ckpt_path and os.path.exists(self.ckpt_path):
            eval_logger.info(f"Loading AIA checkpoint from {self.ckpt_path}")
            finetune_ckpt = torch.load(self.ckpt_path, map_location="cpu")
            self._model.load_state_dict(finetune_ckpt)
            eval_logger.info("AIA checkpoint loaded successfully")

        self._model = self._model.to(self._dtype).to(self._device).eval()
        self._config = self._model.config

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def processor(self):
        return self._processor

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def flatten(self, input_list):
        """Flatten a nested list."""
        output = []
        for item in input_list:
            if isinstance(item, list):
                output.extend(self.flatten(item))
            else:
                output.append(item)
        return output

    def _prepare_images(self, visuals: List) -> List[Image.Image]:
        """Convert various image formats to PIL Images."""
        images = []
        for visual in visuals:
            if isinstance(visual, str):
                visual = Image.open(visual).convert("RGB")
            elif isinstance(visual, Image.Image):
                visual = visual.convert("RGB")
            elif isinstance(visual, dict):
                if "bytes" in visual:
                    from io import BytesIO

                    visual = Image.open(BytesIO(visual["bytes"])).convert("RGB")
                elif "path" in visual:
                    visual = Image.open(visual["path"]).convert("RGB")
                elif "image" in visual:
                    img = visual["image"]
                    if isinstance(img, str):
                        visual = Image.open(img).convert("RGB")
                    elif isinstance(img, Image.Image):
                        visual = img.convert("RGB")
                    else:
                        continue
                else:
                    continue
            elif hasattr(visual, "convert"):
                visual = visual.convert("RGB")
            else:
                continue
            images.append(visual)
        return images

    def understand_image(
        self, prompt: str, images: List[Image.Image], doc_id: str
    ) -> str:
        """
        Understand image and answer question.

        Args:
            prompt: Input text prompt/question
            images: List of PIL Images to understand
            doc_id: Document ID for logging

        Returns:
            Generated text answer
        """
        if not images:
            return ""

        # Create image placeholders
        image_placeholders = "<image_placeholder>\n" * len(images)
        user_content = image_placeholders + prompt

        # Build conversation format
        conversation = [
            {
                "role": "<|User|>",
                "content": user_content,
                "images": images,
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

        # Process inputs
        prepare_inputs = self._processor(
            conversations=conversation, images=images, force_batchify=True
        ).to(self._device, dtype=self._dtype)

        # Get image embeddings
        inputs_embeds = self._model.prepare_inputs_embeds(**prepare_inputs)

        # Generate response
        with torch.no_grad():
            outputs = self._model.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=self._tokenizer.eos_token_id,
                bos_token_id=self._tokenizer.bos_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                use_cache=self.use_cache,
            )

        answer = self._tokenizer.decode(
            outputs[0].cpu().tolist(), skip_special_tokens=True
        )

        del outputs, inputs_embeds, prepare_inputs
        torch.cuda.empty_cache()

        return answer

    @torch.inference_mode()
    def generate_image(
        self, prompt: str, doc_id: str, task: str
    ) -> Tuple[str, List[str]]:
        """
        Generate image from text prompt using classifier-free guidance.

        Args:
            prompt: Input text prompt
            doc_id: Document ID for file naming
            task: Task name for file naming

        Returns:
            Tuple of (empty_text, list_of_image_paths)
        """
        # Build conversation format for generation
        conversation = [
            {"role": "User", "content": prompt},
            {"role": "Assistant", "content": ""},
        ]

        sft_format = self._processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=self._processor.sft_format,
            system_prompt="",
        )
        prompt_with_tag = sft_format + self._processor.image_start_tag

        # Tokenize
        input_ids = self._tokenizer.encode(prompt_with_tag)
        input_ids = torch.LongTensor(input_ids)

        # Prepare tokens for classifier-free guidance (conditional + unconditional)
        tokens = torch.zeros(
            (self.parallel_size * 2, len(input_ids)), dtype=torch.int
        ).to(self._device)
        for i in range(self.parallel_size * 2):
            tokens[i, :] = input_ids
            if i % 2 != 0:
                # Unconditional: mask out the prompt
                tokens[i, 1:-1] = self._processor.pad_id

        inputs_embeds = self._model.language_model.get_input_embeddings()(tokens)
        generated_tokens = torch.zeros(
            (self.parallel_size, self.image_token_num), dtype=torch.int
        ).to(self._device)

        past_key_values = None
        for i in range(self.image_token_num):
            outputs = self._model.language_model.model(
                inputs_embeds=inputs_embeds,
                use_cache=True,
                past_key_values=past_key_values,
            )
            past_key_values = outputs.past_key_values
            hidden_states = outputs.last_hidden_state
            logits = self._model.gen_head(hidden_states[:, -1, :])

            # Classifier-free guidance
            logit_cond = logits[0::2, :]
            logit_uncond = logits[1::2, :]
            logits = logit_uncond + self.cfg_weight * (logit_cond - logit_uncond)

            probs = torch.softmax(logits / self.temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated_tokens[:, i] = next_token.squeeze(dim=-1)

            # Prepare next input
            next_token = torch.cat(
                [next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1
            ).view(-1)
            img_embeds = self._model.prepare_gen_img_embeds(next_token)
            inputs_embeds = img_embeds.unsqueeze(dim=1)

        # Decode generated tokens to image
        dec = self._model.gen_vision_model.decode_code(
            generated_tokens.to(dtype=torch.int),
            shape=[
                self.parallel_size,
                8,
                self.img_size // self.patch_size,
                self.img_size // self.patch_size,
            ],
        )
        dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
        dec = np.clip((dec + 1) / 2 * 255, 0, 255)

        # Save images
        output_images = []
        for i in range(self.parallel_size):
            visual_img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            visual_img[:, :, :] = dec[i]
            safe_filename = f"{task}_{doc_id}_{i}.png"
            image_path = os.path.join(self.output_image_dir, safe_filename)
            PIL.Image.fromarray(visual_img).save(image_path)
            output_images.append(image_path)
            eval_logger.info(f"Saved generated image: {image_path}")

        del generated_tokens, inputs_embeds, past_key_values
        torch.cuda.empty_cache()

        return "", output_images

    def format_output(self, text: str, images: List[str]) -> str:
        """Format output as JSON string."""
        output_dict = {"text": text, "images": images}
        return json.dumps(output_dict, ensure_ascii=False)

    def generate_until(self, requests: List[Instance]) -> List[str]:
        """Main inference method."""
        res = []
        pbar = tqdm(
            total=len(requests),
            disable=(self.rank != 0),
            desc=f"AIA {self.mode.capitalize()}",
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

                # Get images
                visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
                visuals = self.flatten(visuals)
                images = self._prepare_images(visuals)

                if not images:
                    eval_logger.warning(f"No valid images found for doc_id={doc_id}")
                    res.append("")
                    pbar.update(1)
                    continue

                output_text = self.understand_image(prompt, images, str(doc_id))
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
        """Not supported for AIA model."""
        raise NotImplementedError(
            "AIA is a generation model and does not support loglikelihood"
        )

    def generate_until_multi_round(self, requests) -> List[str]:
        """Multi-round dialogue generation (not implemented)."""
        raise NotImplementedError("Multi-round generation not yet implemented for AIA")
