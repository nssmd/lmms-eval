"""
FUDOKI Model - Discrete Flow-based Unified Understanding and Generation

FUDOKI is a unified multimodal model based on discrete flow matching,
supporting both visual understanding and image generation.

Paper: https://arxiv.org/abs/2505.20147
HuggingFace: https://huggingface.co/LucasJinWang/FUDOKI
GitHub: https://github.com/fudoki-hku/FUDOKI

Example usage:
    python -m lmms_eval --model fudoki \
        --model_args pretrained=LucasJinWang/FUDOKI \
        --tasks mme,mmmu_val --batch_size 1 --device cuda:0
"""

import os
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model


# FUDOKI constants
VOCABULARY_SIZE_TXT = 102400
VOCABULARY_SIZE_IMG = 16384
IMG_LEN = 576


def resize_pad(image: Image.Image, image_size: int = 384) -> Image.Image:
    """Resize and pad image to target size while maintaining aspect ratio."""
    w, h = image.size
    if w <= 0 or h <= 0:
        return image.resize((image_size, image_size), Image.Resampling.BILINEAR)

    resize_scale = image_size / max(w, h)
    new_w = max(1, int(w * resize_scale))
    new_h = max(1, int(h * resize_scale))

    padding_color = (127, 127, 127)
    new_image = Image.new("RGB", (image_size, image_size), padding_color)

    if new_w <= 0 or new_h <= 0:
        return image.resize((image_size, image_size), Image.Resampling.BILINEAR)

    image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)

    paste_x = (image_size - new_w) // 2
    paste_y = (image_size - new_h) // 2

    new_image.paste(image, (paste_x, paste_y))
    return new_image


@register_model("fudoki")
class FUDOKI(lmms):
    """
    FUDOKI Model - Discrete Flow-based Unified Multimodal Model

    This model uses discrete flow matching for visual understanding tasks.
    It supports iterative refinement and bidirectional context integration.
    """

    def __init__(
        self,
        pretrained: str = "LucasJinWang/FUDOKI",
        device: str = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = torch.bfloat16,
        batch_size: int = 1,
        trust_remote_code: bool = True,
        # Flow matching parameters
        discrete_fm_steps: int = 64,
        txt_max_length: int = 500,
        cfg_scale: float = 0.0,
        seed: int = 42,
        # Ignored parameters (for compatibility with visual_cot variant)
        save_intermediate: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        self.pretrained = pretrained
        self.discrete_fm_steps = discrete_fm_steps
        self.txt_max_length = txt_max_length
        self.cfg_scale = cfg_scale
        self.seed = seed

        # Set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Determine device - check for distributed setup without initializing Accelerator yet
        # This avoids meta tensor issues during model loading
        if torch.cuda.is_available():
            # Check if running in distributed mode
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            if world_size > 1:
                self._device = torch.device(f"cuda:{local_rank}")
            else:
                self._device = device if isinstance(device, torch.device) else torch.device(device)
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

        # Load FUDOKI model
        eval_logger.info(f"Loading FUDOKI model from {pretrained}")

        try:
            from fudoki.janus.models import VLChatProcessor
            from transformers import AutoModelForCausalLM
        except ImportError:
            raise ImportError(
                "FUDOKI requires the fudoki package. "
                "Please install it from: https://github.com/fudoki-hku/FUDOKI"
            )

        # Load model directly with low_cpu_mem_usage=False to avoid meta tensor issues
        self._model = AutoModelForCausalLM.from_pretrained(
            pretrained,
            trust_remote_code=True,
            low_cpu_mem_usage=False,
            torch_dtype=self._dtype,
        ).to(self._device)
        self._model.eval()

        # Load processor
        self.vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(
            pretrained
        )
        self._tokenizer = self.vl_chat_processor.tokenizer
        self._config = self._model.config

        # Setup embedding paths
        from huggingface_hub import hf_hub_download

        cache_dir = os.path.join(
            os.path.expanduser("~"), ".cache", "huggingface", "hub"
        )

        # Download embedding files if needed
        self.text_embedding_path = hf_hub_download(
            repo_id=pretrained,
            filename="text_embedding.pt",
            cache_dir=cache_dir,
        )
        self.image_embedding_path = hf_hub_download(
            repo_id=pretrained,
            filename="image_embedding.pt",
            cache_dir=cache_dir,
        )

        eval_logger.info(f"Text embedding path: {self.text_embedding_path}")
        eval_logger.info(f"Image embedding path: {self.image_embedding_path}")

        # Initialize flow matching components
        try:
            from fudoki.eval_loop import CFGScaledModel
            from flow_matching.path import MixtureDiscreteSoftmaxProbPath
            from flow_matching.solver import MixtureDiscreteSoftmaxEulerSolver
        except ImportError:
            raise ImportError(
                "FUDOKI requires flow_matching package. "
                "Please install it from: https://github.com/fudoki-hku/FUDOKI"
            )

        self.cfg_weighted_model = CFGScaledModel(
            model=self._model, g_or_u="understanding"
        )
        self.path_txt = MixtureDiscreteSoftmaxProbPath(
            mode="text", embedding_path=self.text_embedding_path
        )
        self.path_img = MixtureDiscreteSoftmaxProbPath(
            mode="image", embedding_path=self.image_embedding_path
        )
        self.solver = MixtureDiscreteSoftmaxEulerSolver(
            model=self.cfg_weighted_model,
            path_txt=self.path_txt,
            path_img=self.path_img,
            vocabulary_size_txt=VOCABULARY_SIZE_TXT,
            vocabulary_size_img=VOCABULARY_SIZE_IMG,
        )

        self.batch_size_per_gpu = int(batch_size)
        assert self.batch_size_per_gpu == 1, "batch_size > 1 not supported for FUDOKI"

        # Setup distributed training (after model is loaded)
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        if world_size > 1:
            accelerator = Accelerator()
            distributed_type_list = [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
                DistributedType.DEEPSPEED,
            ]
            assert accelerator.distributed_type in distributed_type_list, (
                "Unsupported distributed type. Only DDP, FSDP, and DeepSpeed supported"
            )
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(
                    f"Using {accelerator.num_processes} devices with parallelism"
                )
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            eval_logger.info(f"Using single device: {self._device}")
            self._rank = 0
            self._world_size = 1

        eval_logger.info("FUDOKI model initialized successfully")

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

    def flatten(self, input_list: List) -> List:
        """Flatten a nested list."""
        new_list = []
        for i in input_list:
            for j in i:
                new_list.append(j)
        return new_list

    def _extract_image(self, visual) -> Optional[Image.Image]:
        """Extract PIL Image from various formats."""
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

    def _prepare_conversation(self, context: str, has_image: bool) -> list:
        """Prepare conversation format for FUDOKI."""
        if has_image:
            content = "<image_placeholder>" + context
        else:
            content = context

        conversation = [
            {"role": "User", "content": content},
            {"role": "Assistant", "content": ""},
        ]
        return conversation

    def _generate_response(
        self,
        context: str,
        image: Optional[Image.Image] = None,
    ) -> str:
        """Generate response using discrete flow matching."""
        from torchvision import transforms

        # Prepare conversation
        has_image = image is not None
        conversation = self._prepare_conversation(context, has_image)

        # Apply SFT template
        sft_format = self.vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=self.vl_chat_processor.sft_format,
            system_prompt=self.vl_chat_processor.system_prompt,
        )

        # Process image if present
        if has_image and "<image_placeholder>" in sft_format:
            transform = transforms.Compose(
                [
                    transforms.Lambda(resize_pad),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True
                    ),
                ]
            )
            img_tensor = transform(image)
            img_len = IMG_LEN
        else:
            img_tensor = None
            img_len = IMG_LEN

        # Tokenize
        input_ids = self.vl_chat_processor.tokenizer.encode(sft_format)
        input_ids = torch.LongTensor(input_ids)

        # Add image tokens if needed
        if has_image:
            image_token_mask = input_ids == self.vl_chat_processor.image_id
            image_indices = image_token_mask.nonzero()
            input_ids, _ = self.vl_chat_processor.add_image_token(
                image_indices=image_indices,
                input_ids=input_ids,
            )

        # Pad tokens
        original_input_id_len = input_ids.shape[0]
        answer_token_num = self.txt_max_length  # Maximum answer length

        if original_input_id_len >= self.txt_max_length + img_len:
            eval_logger.warning(f"Input too long ({original_input_id_len}), truncating to {self.txt_max_length + img_len}...")
            input_ids = input_ids[: self.txt_max_length + img_len]
            original_input_id_len = input_ids.shape[0]

        rows_to_pad = self.txt_max_length + img_len - input_ids.shape[0]
        input_ids = torch.cat(
            [
                input_ids,
                torch.LongTensor([self.vl_chat_processor.pad_id]).repeat(rows_to_pad),
            ],
            dim=0,
        )

        # Attention mask: True for input + answer area, False for padding beyond
        attention_mask = torch.zeros((input_ids.shape[0]), dtype=torch.bool)
        attention_mask[:original_input_id_len + answer_token_num] = True

        # Obtain image token mask
        if has_image:
            image_expanded_token_mask = (
                input_ids == self.vl_chat_processor.image_id
            ).to(dtype=torch.int)
            image_expanded_mask_indices = torch.where(image_expanded_token_mask == 1)[0]
            input_ids[image_expanded_mask_indices] = 0
        else:
            image_expanded_token_mask = torch.zeros_like(input_ids)

        # Obtain text token mask (positions after "Assistant:" up to answer_token_num)
        text_expanded_token_mask = torch.zeros_like(image_expanded_token_mask)
        split_token = self.vl_chat_processor.tokenizer.encode(
            "Assistant:", add_special_tokens=False
        )
        split_token_length = len(split_token)

        start_index = -1
        for j in range(len(input_ids) - split_token_length + 1):
            if input_ids[j : j + split_token_length].numpy().tolist() == split_token:
                start_index = j
                break

        if start_index != -1:
            # Only mark from after "Assistant:" to original_input_id_len + answer_token_num
            text_expanded_token_mask[(start_index + split_token_length):(original_input_id_len + answer_token_num)] = 1
        else:
            eval_logger.warning("Split token 'Assistant:' not found in input_ids")
            # Fallback: mark from original_input_id_len to original_input_id_len + answer_token_num
            text_expanded_token_mask[original_input_id_len:(original_input_id_len + answer_token_num)] = 1

        # Build data_info dictionary
        generation_or_understanding_mask = 0  # Understanding mode
        data_info = {}
        data_info["text_token_mask"] = (
            text_expanded_token_mask.unsqueeze(0).to(self._device)
        )
        data_info["image_token_mask"] = (
            image_expanded_token_mask.unsqueeze(0).to(self._device)
        )
        data_info["generation_or_understanding_mask"] = (
            torch.Tensor([generation_or_understanding_mask])
            .unsqueeze(0)
            .to(self._device)
            .to(dtype=torch.int)
        )
        data_info["attention_mask"] = attention_mask.unsqueeze(0).to(self._device)
        data_info["sft_format"] = sft_format

        if has_image and img_tensor is not None:
            data_info["understanding_img"] = img_tensor.unsqueeze(0).to(
                device=self._device, dtype=self._dtype
            )
            data_info["has_understanding_img"] = (
                torch.Tensor([True]).to(dtype=torch.int).unsqueeze(0).to(self._device)
            )
        else:
            data_info["understanding_img"] = (
                torch.zeros((3, 384, 384)).unsqueeze(0).to(
                    device=self._device, dtype=self._dtype
                )
            )
            data_info["has_understanding_img"] = (
                torch.Tensor([False]).to(dtype=torch.int).unsqueeze(0).to(self._device)
            )

        input_ids = input_ids.unsqueeze(0).to(self._device)

        # Initialize random tokens for flow matching
        x_0_txt = torch.randint(
            VOCABULARY_SIZE_TXT, input_ids.shape, dtype=torch.long, device=self._device
        )
        x_init = x_0_txt * data_info["text_token_mask"] + input_ids * (
            1 - data_info["text_token_mask"]
        )

        # Run flow matching solver
        with torch.no_grad():
            synthetic_samples = self.solver.sample(
                x_init=x_init,
                step_size=1.0 / self.discrete_fm_steps,
                verbose=False,
                return_intermediates=False,
                div_free=0,
                dtype_categorical=torch.float32,
                datainfo=data_info,
                cfg_scale=self.cfg_scale,
            )

        # Decode only the answer portion (like official code)
        answer_tokens = synthetic_samples[0, :answer_token_num]
        full_response = self.vl_chat_processor.tokenizer.decode(answer_tokens)

        # Remove content after EOS token
        def keep_strictly_before_eos(text, eos_token="<｜end▁of▁sentence｜>"):
            idx = text.find(eos_token)
            if idx == -1:
                return text
            return text[:idx]

        response = keep_strictly_before_eos(full_response)

        # Extract after "Assistant:" if present
        if "Assistant:" in response:
            parts = response.rsplit("Assistant:", 1)
            if len(parts) > 1:
                response = parts[1].strip()

        # Clean up
        del synthetic_samples, x_init, input_ids
        torch.cuda.empty_cache()

        return response

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """Compute log-likelihood (not implemented for FUDOKI)."""
        raise NotImplementedError("Loglikelihood not implemented for FUDOKI")

    def generate_until(self, requests: List[Instance]) -> List[str]:
        """Generate text until stopping criteria are met."""
        res = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(
            total=len(requests),
            disable=(self.rank != 0),
            desc="FUDOKI Generating",
        )

        # Group requests by generation kwargs
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

            # Get visuals
            visuals = [
                doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id
            ]
            visuals = self.flatten(visuals)

            # Get generation kwargs (for compatibility, though FUDOKI uses flow matching)
            gen_kwargs = all_gen_kwargs[0]

            assert len(contexts) == 1, "Batch size must be 1"
            context = contexts[0]

            # Extract image - FUDOKI only supports single image, use the last one (test image)
            image = None
            if visuals:
                image = self._extract_image(visuals[-1])
                if image is not None:
                    eval_logger.debug(f"Processing image: {image.size} (using last of {len(visuals)} images)")

            # Generate response
            response = self._generate_response(context, image)

            res.append(response)
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), response)
            pbar.update(1)

        # Reorder results to original order
        res = re_ords.get_original(res)
        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        """Generate for multi-round conversations (not implemented)."""
        raise NotImplementedError("Multi-round generation not yet implemented")
