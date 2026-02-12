import torch
import sys
import os
from typing import List, Optional, Tuple, Union
from datetime import timedelta

from accelerate import Accelerator, InitProcessGroupKwargs
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

from loguru import logger as eval_logger

# Add LLaDA path to sys.path
LLADA_PATH = "/home/aiscuser/LLaDA"
if LLADA_PATH not in sys.path:
    sys.path.insert(0, LLADA_PATH)

try:
    from generate import generate
    from get_log_likelihood import get_log_likelihood
    from transformers import AutoTokenizer, AutoModel
except Exception as e:
    eval_logger.error(f"Failed to import LLaDA dependencies: {e}")


@register_model("llada")
class LLaDA(lmms):
    """
    LLaDA Model - A diffusion-based language model
    """

    def __init__(
        self,
        pretrained: str = "GSAI-ML/LLaDA-8B-Instruct",
        device: Optional[str] = "cuda:0",
        batch_size: Optional[Union[int, str]] = 1,
        gen_length: int = 128,
        max_gen_length: Optional[int] = None,
        steps: int = 128,
        block_length: int = 32,
        temperature: float = 0.0,
        cfg_scale: float = 0.0,
        remasking: str = "low_confidence",
        mc_num: int = 128,
        mc_batch_size: int = 16,
        mask_id: int = 126336,
        **kwargs,
    ) -> None:
        super().__init__()

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        self.accelerator = accelerator

        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
        else:
            self._device = torch.device(device)

        self._model = None
        self._tokenizer = None
        self.pretrained = pretrained
        self.gen_length = gen_length
        self.max_gen_length = max_gen_length if max_gen_length is not None else gen_length
        self.steps = steps
        self.block_length = block_length
        self.temperature = temperature
        self.cfg_scale = cfg_scale
        self.remasking = remasking
        self.mc_num = mc_num
        self.mc_batch_size = mc_batch_size
        self.mask_id = mask_id
        self._batch_size = int(batch_size)

        eval_logger.info(f"Initializing LLaDA model: {pretrained}")
        eval_logger.info(f"Generation parameters: gen_length={gen_length}, max_gen_length={self.max_gen_length}, steps={steps}, block_length={block_length}")

    @property
    def model(self):
        if self._model is None:
            eval_logger.info(f"Loading LLaDA model from {self.pretrained}")
            self._model = AutoModel.from_pretrained(
                self.pretrained,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16
            ).to(self._device).eval()
        return self._model

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            eval_logger.info(f"Loading tokenizer from {self.pretrained}")
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.pretrained,
                trust_remote_code=True
            )
        return self._tokenizer

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self.accelerator.local_process_index

    @property
    def world_size(self):
        return self.accelerator.num_processes

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """
        Compute log-likelihood of generating a continuation from a context.
        Uses Monte Carlo estimation for diffusion-based model.
        """
        results = []

        for instance in tqdm(requests, desc="Computing log-likelihood"):
            context, continuation = instance.args

            # Tokenize context and continuation
            context_tokens = torch.tensor(
                self.tokenizer(context)['input_ids']
            ).to(self.device)

            continuation_tokens = torch.tensor(
                self.tokenizer(continuation)['input_ids']
            ).to(self.device)

            # Compute log-likelihood using Monte Carlo estimation
            log_likelihood = get_log_likelihood(
                self.model,
                context_tokens,
                continuation_tokens,
                mc_num=self.mc_num,
                batch_size=self.mc_batch_size,
                cfg_scale=self.cfg_scale,
                mask_id=self.mask_id
            )

            # For diffusion models, we don't have a greedy baseline
            # so we always return False for is_greedy
            results.append((log_likelihood, False))

        return results

    def generate_until(self, requests: List[Instance]) -> List[str]:
        """
        Generate text until stopping criteria are met.
        Uses LLaDA's diffusion-based generation.
        """
        results = []

        for instance in tqdm(requests, desc="Generating"):
            context = instance.args[0]
            generation_kwargs = instance.args[1] if len(instance.args) > 1 else {}

            # Apply chat template if context is a conversation
            if isinstance(context, list):
                # Multi-turn conversation
                context_text = self.tokenizer.apply_chat_template(
                    context,
                    add_generation_prompt=True,
                    tokenize=False
                )
            else:
                # Single turn - wrap in chat format
                messages = [{"role": "user", "content": context}]
                context_text = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False
                )

            # Tokenize input
            input_ids = torch.tensor(
                self.tokenizer(context_text)['input_ids']
            ).to(self.device).unsqueeze(0)

            # Get generation parameters
            gen_length = generation_kwargs.get('max_new_tokens', self.gen_length)
            temperature = generation_kwargs.get('temperature', self.temperature)

            # Limit gen_length to max_gen_length if specified
            if gen_length > self.max_gen_length:
                eval_logger.warning(f"Task requested gen_length={gen_length}, but limiting to max_gen_length={self.max_gen_length}")
                gen_length = self.max_gen_length

            # Ensure gen_length is a multiple of block_length
            if gen_length % self.block_length != 0:
                gen_length = ((gen_length // self.block_length) + 1) * self.block_length
                eval_logger.warning(f"Adjusted gen_length to {gen_length} (must be multiple of block_length={self.block_length})")

            # Generate using LLaDA's diffusion-based generation
            output = generate(
                self.model,
                input_ids,
                steps=self.steps,
                gen_length=gen_length,
                block_length=self.block_length,
                temperature=temperature,
                cfg_scale=self.cfg_scale,
                remasking=self.remasking,
                mask_id=self.mask_id
            )

            # Decode only the generated part
            generated_text = self.tokenizer.batch_decode(
                output[:, input_ids.shape[1]:],
                skip_special_tokens=True
            )[0]

            results.append(generated_text)

        return results

    def generate_until_multi_round(self, requests: List[Instance]) -> List[str]:
        """
        Generate text for multi-round conversations.
        Similar to generate_until but handles conversation history.
        """
        results = []

        for instance in tqdm(requests, desc="Generating (multi-round)"):
            context = instance.args[0]
            generation_kwargs = instance.args[1] if len(instance.args) > 1 else {}

            # Context should be a list of messages for multi-round
            if isinstance(context, list):
                messages = context
            else:
                # Fallback to single turn
                messages = [{"role": "user", "content": context}]

            # Apply chat template
            context_text = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )

            # Tokenize input
            input_ids = torch.tensor(
                self.tokenizer(context_text)['input_ids']
            ).to(self.device).unsqueeze(0)

            # Get generation parameters
            gen_length = generation_kwargs.get('max_new_tokens', self.gen_length)
            temperature = generation_kwargs.get('temperature', self.temperature)

            # Limit gen_length to max_gen_length if specified
            if gen_length > self.max_gen_length:
                eval_logger.warning(f"Task requested gen_length={gen_length}, but limiting to max_gen_length={self.max_gen_length}")
                gen_length = self.max_gen_length

            # Ensure gen_length is a multiple of block_length
            if gen_length % self.block_length != 0:
                gen_length = ((gen_length // self.block_length) + 1) * self.block_length
                eval_logger.warning(f"Adjusted gen_length to {gen_length} (must be multiple of block_length={self.block_length})")

            # Generate using LLaDA's diffusion-based generation
            output = generate(
                self.model,
                input_ids,
                steps=self.steps,
                gen_length=gen_length,
                block_length=self.block_length,
                temperature=temperature,
                cfg_scale=self.cfg_scale,
                remasking=self.remasking,
                mask_id=self.mask_id
            )

            # Decode only the generated part
            generated_text = self.tokenizer.batch_decode(
                output[:, input_ids.shape[1]:],
                skip_special_tokens=True
            )[0]

            results.append(generated_text)

        return results

