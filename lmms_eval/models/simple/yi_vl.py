import os
import sys
import warnings
from datetime import timedelta
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from loguru import logger as eval_logger
from packaging import version
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

warnings.filterwarnings("ignore")

torch.backends.cuda.matmul.allow_tf32 = True

# Add Yi-VL repository to Python path
# Expected: lmms-eval/Yi/VL/ directory at project root
wd = Path(__file__).parent.parent.parent.parent.resolve()
yi_vl_path = os.path.join(str(wd), "Yi", "VL")
if os.path.exists(yi_vl_path):
    sys.path.insert(0, yi_vl_path)
    eval_logger.info(f"Added Yi-VL path to sys.path: {yi_vl_path}")
else:
    eval_logger.warning(
        f"Yi-VL repository not found at {yi_vl_path}. "
        f"Please clone it: cd {wd} && git clone https://github.com/01-ai/Yi.git"
    )

# Import Yi-VL dependencies
try:
    from llava.conversation import conv_templates
    from llava.mm_utils import (
        KeywordsStoppingCriteria,
        expand2square,
        get_model_name_from_path,
        load_pretrained_model,
        tokenizer_image_token,
    )
    from llava.model.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, key_info

    YI_VL_AVAILABLE = True
except Exception as e:
    YI_VL_AVAILABLE = False
    eval_logger.debug(
        "Yi-VL is not installed. Please install Yi-VL to use this model.\nError: %s" % e
    )
    # Define dummy variables to avoid NameError
    DEFAULT_IMAGE_TOKEN = "<image_placeholder>"
    IMAGE_TOKEN_INDEX = -200
    key_info = {"model_path": None}
    conv_templates = None

# Determine best attention implementation
if version.parse(torch.__version__) >= version.parse("2.1.2"):
    best_fit_attn_implementation = "sdpa"
else:
    best_fit_attn_implementation = "eager"


@register_model("yi_vl")
class YiVL(lmms):
    """
    Yi Vision Language Model (Yi-VL)

    Yi-VL is a multimodal model based on LLaVA architecture that combines:
    - Vision Transformer (ViT): CLIP ViT-H/14 for image encoding
    - Projection Module: Two-layer MLP with layer normalization
    - Large Language Model: Yi-6B-Chat for text understanding

    Supports both English and Chinese for visual question answering and image description.

    Example usage:
    accelerate launch -m lmms_eval \
        --model yi_vl \
        --model_args pretrained=01-ai/Yi-VL-6B \
        --tasks mmbench \
        --batch_size 1 \
        --output_path ./logs/
    """

    def __init__(
        self,
        pretrained: str = "01-ai/Yi-VL-6B",
        truncation: Optional[bool] = True,
        device: Optional[str] = "cuda:0",
        batch_size: Optional[Union[int, str]] = 1,
        attn_implementation: Optional[str] = best_fit_attn_implementation,
        device_map: str = "auto",
        conv_template: str = "mm_default",
        use_cache: bool = True,
        truncate_context: bool = False,
        temperature: float = 0.2,
        top_p: Optional[float] = None,
        num_beams: int = 1,
        max_new_tokens: int = 1024,
        **kwargs,
    ) -> None:
        super().__init__()

        # Check if Yi-VL is available
        if not YI_VL_AVAILABLE:
            raise ImportError(
                "Yi-VL dependencies are not available. "
                "Please ensure:\n"
                "  1. Yi repository is cloned at lmms-eval root: "
                "git clone https://github.com/01-ai/Yi.git\n"
                "  2. Yi-VL requirements are installed: "
                "cd Yi/VL && pip install -r requirements.txt"
            )

        # Validate kwargs
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        # Setup accelerator
        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        self.accelerator = accelerator

        # Setup device configuration
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        # Store model parameters
        self.pretrained = pretrained
        self.truncation = truncation
        self.batch_size_per_gpu = int(batch_size)
        self.conv_template = conv_template
        self.use_cache = use_cache
        self.truncate_context = truncate_context
        self.temperature = temperature
        self.top_p = top_p
        self.num_beams = num_beams
        self.max_new_tokens = max_new_tokens

        # Load Yi-VL model
        eval_logger.info(f"Loading Yi-VL model from {pretrained}")
        self._load_model()

        # Setup distributed training if needed
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
                DistributedType.DEEPSPEED,
            ], "Unsupported distributed type provided. Only DDP, FSDP and DEEPSPEED are supported."

            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs_deepspeed = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(
                    must_match=True, **kwargs_deepspeed
                )
                eval_logger.info(
                    "Detected that you are using DistributedType.DEEPSPEED. "
                    "Make sure you run `accelerate config` and set zero stage to 0"
                )

            if accelerator.distributed_type == DistributedType.FSDP or accelerator.distributed_type == DistributedType.DEEPSPEED:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)

            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        elif accelerator.num_processes == 1 and device_map == "auto":
            eval_logger.info(f"Using {accelerator.num_processes} devices with tensor parallelism")
            self._rank = 0
            self._world_size = 1
        else:
            eval_logger.info(f"Using single device: {self._device}")
            self.model.to(self._device)
            self._rank = 0
            self._world_size = 1

        eval_logger.info("Yi-VL model initialized successfully")

    def _load_model(self):
        """Load Yi-VL model components"""
        model_path = os.path.expanduser(self.pretrained)

        # Set model path in key_info for Yi-VL internal use
        key_info["model_path"] = model_path

        # Get model name
        get_model_name_from_path(model_path)

        # Load pretrained model
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path, device_map=self.device_map
        )

        # Store model components
        self._tokenizer = tokenizer
        self._model = model
        self._image_processor = image_processor
        self._max_length = context_len
        self._config = model.config

        # Set model to eval mode
        self.model.eval()
        self.model.tie_weights()

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

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

    def pad_sequence(self, input_ids, batch_first, padding_value):
        """Pad sequence with proper handling of padding side"""
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=batch_first, padding_value=padding_value
        )
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None) -> List[int]:
        """Tokenize string"""
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_decode(self, tokens):
        """Decode tokens to string"""
        return self.tokenizer.decode(tokens)

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
        """Main inference method for Yi-VL"""
        res = []
        pbar = tqdm(
            total=len(requests),
            disable=(self.rank != 0),
            desc="Yi-VL Generating"
        )

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [
            reg.args for reg in requests
        ]:
            eval_logger.info(f"[DEBUG] Processing doc_id={doc_id}, task={task}")

            # Get visual data
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)

            eval_logger.info(f"[DEBUG] doc_id={doc_id}: visuals count={len(visuals) if visuals else 0}")
            if visuals and len(visuals) > 0:
                for idx, img in enumerate(visuals):
                    eval_logger.info(f"[DEBUG] doc_id={doc_id}: image[{idx}] type={type(img)}, mode={img.mode if hasattr(img, 'mode') else 'N/A'}, size={img.size if hasattr(img, 'size') else 'N/A'}")

            if not visuals or len(visuals) == 0:
                eval_logger.warning(f"[DEBUG] No visual data found for doc_id={doc_id}")
                res.append("")
                pbar.update(1)
                continue

            # Prepare prompt with image tokens for all images
            question = contexts
            # Add one image token per image
            num_images = len(visuals)
            eval_logger.info(f"[DEBUG] doc_id={doc_id}: Processing {num_images} images, question length={len(question)}")

            # Handle multiple images: concatenate them into one image
            if num_images > 1:
                eval_logger.info(f"[DEBUG] doc_id={doc_id}: Concatenating {num_images} images into one")
                # Concatenate images horizontally
                from PIL import Image

                # Get max height and total width
                max_height = max(img.size[1] for img in visuals)
                total_width = sum(img.size[0] for img in visuals)

                # Create new image
                concatenated = Image.new('RGB', (total_width, max_height), (255, 255, 255))

                # Paste images
                x_offset = 0
                for img in visuals:
                    # Convert to RGB if needed
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    concatenated.paste(img, (x_offset, 0))
                    x_offset += img.size[0]

                # Replace visuals with concatenated image
                visuals = [concatenated]
                eval_logger.info(f"[DEBUG] doc_id={doc_id}: Concatenated image size={concatenated.size}")

            # Check if question already contains image tokens
            if DEFAULT_IMAGE_TOKEN in question or "<image>" in question:
                # Question already has image tokens, use it directly
                eval_logger.info(f"[DEBUG] doc_id={doc_id}: Question already contains image tokens, using as-is")
                prompt_with_image = question
            else:
                # Add single image token (since we concatenated multiple images into one)
                prompt_with_image = DEFAULT_IMAGE_TOKEN + "\n" + question

            eval_logger.info(f"[DEBUG] doc_id={doc_id}: prompt_with_image length={len(prompt_with_image)}")

            # Setup conversation
            conv = conv_templates[self.conv_template].copy()
            conv.append_message(conv.roles[0], prompt_with_image)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            eval_logger.info(f"[DEBUG] doc_id={doc_id}: conv_template={self.conv_template}, conv.sep='{conv.sep}'")
            eval_logger.info(f"[DEBUG] doc_id={doc_id}: Final prompt length={len(prompt)}, first 2000 chars:\n{prompt[:2000]}")

            # Tokenize prompt
            input_ids = (
                tokenizer_image_token(
                    prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                )
                .unsqueeze(0)
                .to(self.device)
            )

            eval_logger.info(f"[DEBUG] doc_id={doc_id}: input_ids shape={input_ids.shape}")
            eval_logger.info(f"[DEBUG] doc_id={doc_id}: IMAGE_TOKEN_INDEX={IMAGE_TOKEN_INDEX}")
            # 检查 input_ids 中有多少个 IMAGE_TOKEN_INDEX
            image_token_count = (input_ids[0] == IMAGE_TOKEN_INDEX).sum().item()
            eval_logger.info(f"[DEBUG] doc_id={doc_id}: Number of IMAGE_TOKEN in input_ids: {image_token_count}")

            # Process all images (now should be only 1 after concatenation)
            image_tensors = []
            for idx, image in enumerate(visuals):
                eval_logger.info(f"[DEBUG] doc_id={doc_id}: Processing image[{idx}], mode={image.mode}, size={image.size}")

                # Convert to RGB if needed (handle RGBA, L, etc.)
                if image.mode != 'RGB':
                    eval_logger.info(f"[DEBUG] doc_id={doc_id}: Converting image[{idx}] from {image.mode} to RGB")
                    image = image.convert('RGB')

                if getattr(self.model.config, "image_aspect_ratio", None) == "pad":
                    image = expand2square(
                        image,
                        tuple(int(x * 255) for x in self._image_processor.image_mean)
                    )

                try:
                    image_tensor = self._image_processor.preprocess(
                        image, return_tensors="pt"
                    )["pixel_values"][0]
                    image_tensors.append(image_tensor)
                    eval_logger.info(f"[DEBUG] doc_id={doc_id}: Image[{idx}] preprocessed successfully, tensor shape={image_tensor.shape}")
                except Exception as e:
                    eval_logger.error(f"[DEBUG] doc_id={doc_id}: Failed to preprocess image[{idx}]: {e}")
                    raise

            # Stack all image tensors (should be only 1 after concatenation)
            if len(image_tensors) == 0:
                eval_logger.error(f"[DEBUG] doc_id={doc_id}: No image tensors to stack! This should not happen.")
                res.append("")
                pbar.update(1)
                continue

            # For Yi-VL, we need shape [1, 3, H, W] for single image
            if len(image_tensors) == 1:
                image_tensors = image_tensors[0].unsqueeze(0)
            else:
                image_tensors = torch.stack(image_tensors)

            eval_logger.info(f"[DEBUG] doc_id={doc_id}: Final image tensors shape={image_tensors.shape}")

            # Setup stopping criteria
            stop_str = conv.sep
            eval_logger.info(f"[DEBUG] doc_id={doc_id}: stop_str='{stop_str}', stop_str length={len(stop_str)}")
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(
                keywords, self.tokenizer, input_ids
            )

            # Create attention mask
            attention_mask = torch.ones_like(input_ids)

            # Generate response
            with torch.inference_mode():
                try:
                    eval_logger.info(f"[DEBUG] doc_id={doc_id}: Starting generation - input_ids shape={input_ids.shape}, images shape={image_tensors.shape}, max_new_tokens={self.max_new_tokens}")
                    output_ids = self.model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        images=image_tensors.to(dtype=torch.bfloat16).to(self.device),
                        do_sample=True if self.temperature > 0 else False,
                        temperature=self.temperature if self.temperature > 0 else None,
                        top_p=self.top_p,
                        num_beams=self.num_beams,
                        stopping_criteria=[stopping_criteria],
                        max_new_tokens=self.max_new_tokens,
                        use_cache=self.use_cache,
                    )
                    eval_logger.info(f"[DEBUG] doc_id={doc_id}: Generation completed - output_ids shape={output_ids.shape}")
                except Exception as e:
                    eval_logger.error(f"[DEBUG] Generation failed for doc_id={doc_id}: {e}")
                    import traceback
                    eval_logger.error(f"[DEBUG] Traceback: {traceback.format_exc()}")
                    res.append("")
                    pbar.update(1)
                    continue

            # Decode output
            input_token_len = input_ids.shape[1]
            n_diff_input_output = (
                (input_ids != output_ids[:, :input_token_len]).sum().item()
            )
            if n_diff_input_output > 0:
                eval_logger.debug(
                    f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
                )

            outputs = self.tokenizer.batch_decode(
                output_ids[:, input_token_len:], skip_special_tokens=True
            )[0]
            outputs = outputs.strip()

            eval_logger.info(f"[DEBUG] doc_id={doc_id}: Decoded output length={len(outputs)}, first 500 chars: {outputs[:500] if outputs else '(EMPTY)'}")

            # Remove stop string if present
            if outputs.endswith(stop_str):
                outputs = outputs[: -len(stop_str)]
            outputs = outputs.strip()

            eval_logger.info(f"[DEBUG] doc_id={doc_id}: Final output length={len(outputs)}, is_empty={len(outputs)==0}")

            res.append(outputs)
            pbar.update(1)

        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """Not implemented for Yi-VL"""
        raise NotImplementedError(
            "Yi-VL is a generation model and does not support loglikelihood"
        )

    def generate_until_multi_round(self, requests) -> List[str]:
        """Multi-round dialogue generation"""
        raise NotImplementedError(
            "TODO: Implement multi-round dialogue generation for Yi-VL"
        )
