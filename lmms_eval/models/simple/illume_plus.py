import os
from typing import List, Optional, Tuple, Union
import re
import time
import sys

import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model, MODEL_REGISTRY

# Prevent duplicate registration
if "illume_plus" in MODEL_REGISTRY:
    eval_logger.warning("illume_plus already registered, skipping re-registration")
    # Remove the old registration to allow re-registration
    del MODEL_REGISTRY["illume_plus"]


@register_model("illume_plus")
class ILLUMEPlus(lmms):
    """
    ILLUME+: Unified multimodal understanding and generation model.
    https://huggingface.co/ILLUME-MLLM/illume_plus-qwen2_5-7b-hf
    """

    def __init__(
        self,
        pretrained: str = "ILLUME-MLLM/illume_plus-qwen2_5-7b-hf",
        device: str = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "bfloat16",
        batch_size: int = 1,
        trust_remote_code: Optional[bool] = True,
        use_cache: bool = True,
        max_new_tokens: int = 2048,
        attn_implementation: Optional[str] = "sdpa",
        device_map: Optional[str] = None,
        infer_auto_device_map: bool = False,
        save_intermediate: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        self.infer_auto_device_map = infer_auto_device_map
        self.save_intermediate = save_intermediate
        self.device_map = device_map

        self.pretrained = pretrained
        self.max_new_tokens = max_new_tokens
        self.use_cache = use_cache
        self.trust_remote_code = trust_remote_code

        # Determine if we should use multi-GPU with device_map="auto"
        # When device_map is set, we don't use Accelerator for distribution
        self._use_device_map = device_map is not None

        if self._use_device_map:
            # Use device_map for multi-GPU (model parallelism)
            eval_logger.info(f"Using device_map={device_map} for multi-GPU model parallelism")
            self._device = torch.device("cuda:0")  # Primary device for inputs
            self._use_accelerator = False
            self._accelerator = None
        else:
            # Use Accelerator for data parallelism (multi-process)
            eval_logger.info("=" * 80)
            eval_logger.info("DEBUG: About to initialize Accelerator")
            eval_logger.info(f"DEBUG: RANK = {os.environ.get('RANK', 'not set')}")
            eval_logger.info(f"DEBUG: WORLD_SIZE = {os.environ.get('WORLD_SIZE', 'not set')}")
            eval_logger.info(f"DEBUG: LOCAL_RANK = {os.environ.get('LOCAL_RANK', 'not set')}")
            eval_logger.info("=" * 80)

            try:
                accelerator = Accelerator()
                eval_logger.info(f"DEBUG: Accelerator initialized, num_processes = {accelerator.num_processes}")
                if accelerator.num_processes > 1:
                    self._device = torch.device(f"cuda:{accelerator.local_process_index}")
                    self._use_accelerator = True
                    self._accelerator = accelerator
                else:
                    self._device = torch.device(device) if isinstance(device, str) else device
                    self._use_accelerator = False
                    self._accelerator = None
            except Exception as e:
                eval_logger.warning(f"Accelerator initialization failed: {e}, using single device mode")
                self._device = torch.device(device) if isinstance(device, str) else device
                self._use_accelerator = False
                self._accelerator = None

        # Determine dtype
        if dtype == "bfloat16" or dtype == "bf16":
            self._dtype = torch.bfloat16
        elif dtype == "float16" or dtype == "fp16":
            self._dtype = torch.float16
        elif dtype == "float32" or dtype == "fp32":
            self._dtype = torch.float32
        else:
            self._dtype = torch.bfloat16

        # Load model
        eval_logger.info(f"Loading ILLUME+ model from {pretrained}")
        eval_logger.info(f"_use_device_map={self._use_device_map}, device_map={self.device_map}")
        self._load_model(pretrained, attn_implementation)

        self.batch_size_per_gpu = int(batch_size)
        assert self.batch_size_per_gpu == 1, "batch_size > 1 not supported for ILLUME+"

        # Setup distributed training if using accelerator (not device_map)
        # IMPORTANT: Do not use Accelerator when device_map is set
        if self._use_device_map:
            eval_logger.info("Skipping Accelerator setup because device_map is being used")
            self._rank = 0
            self._world_size = 1
        elif self._use_accelerator and self._accelerator.num_processes > 1:
            distributed_type_list = [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
                DistributedType.DEEPSPEED,
            ]
            assert self._accelerator.distributed_type in distributed_type_list, (
                "Unsupported distributed type. Only DDP, FSDP, and DeepSpeed supported"
            )
            if self._accelerator.distributed_type == DistributedType.FSDP:
                self._model = self._accelerator.prepare(self._model)
            else:
                self._model = self._accelerator.prepare_model(
                    self._model, evaluation_mode=True
                )
            self.accelerator = self._accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(
                    f"Using {self._accelerator.num_processes} devices with parallelism"
                )
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            eval_logger.info(f"Using single device: {self._device}")
            self._rank = 0
            self._world_size = 1

        eval_logger.info(f"Final setup: rank={self._rank}, world_size={self._world_size}, use_device_map={self._use_device_map}")
        eval_logger.info("ILLUME+ model initialized successfully")

    def _load_model(self, pretrained: str, attn_implementation: Optional[str]):
        """Load ILLUME+ model and processor."""
        try:
            from transformers import AutoModel, AutoProcessor

            # Bypass torch.load security check for .bin files
            os.environ["TRANSFORMERS_ALLOW_UNSAFE_LOAD"] = "1"

            eval_logger.info("Loading ILLUME+ model with transformers")

            # Check if model path exists
            if os.path.exists(pretrained):
                eval_logger.info(f"Loading from local path: {pretrained}")
            else:
                eval_logger.info(f"Loading from HuggingFace Hub: {pretrained}")

            # Load processor first (lightweight)
            eval_logger.info(f"Loading processor from {pretrained}")
            
            processor_kwargs = {
                "trust_remote_code": self.trust_remote_code,
            }

            eval_logger.info("=" * 80)
            eval_logger.info("DEBUG: About to call AutoProcessor.from_pretrained")
            sys.stdout.flush()

            # Set environment variable to avoid potential hangs in custom code
            os.environ["TOKENIZERS_PARALLELISM"] = "false"

            start_time = time.time()
            self._processor = AutoProcessor.from_pretrained(
                pretrained, **processor_kwargs
            )
            elapsed = time.time() - start_time
            eval_logger.info(f"Processor loaded in {elapsed:.1f} seconds")

            self._tokenizer = self._processor.tokenizer
            eval_logger.info("Processor loaded successfully")

            # Load model with proper device handling
            eval_logger.info(f"Loading model from {pretrained} to {self._device}")

            # Determine device_map strategy
            if self.infer_auto_device_map:
                final_device_map = "auto"
                eval_logger.info("Using infer_auto_device_map for multi-GPU model parallelism")
            elif self.device_map is not None:
                final_device_map = self.device_map
                eval_logger.info(f"Using user-specified device_map: {final_device_map}")
            else:
                final_device_map = self._device
                eval_logger.info(f"Using single device: {final_device_map}")

            model_kwargs = {
                "torch_dtype": self._dtype,
                "low_cpu_mem_usage": True,
                "trust_remote_code": self.trust_remote_code,
                "device_map": final_device_map,
            }

            # Add memory optimization when using device_map="auto"
            if isinstance(final_device_map, str) and final_device_map in ["auto", "balanced", "balanced_low_0", "sequential"]:
                model_kwargs["max_memory"] = {i: "38GiB" for i in range(torch.cuda.device_count())}
                eval_logger.info(f"Setting max_memory per GPU to 38GiB for {torch.cuda.device_count()} GPUs")

            if attn_implementation is not None:
                model_kwargs["attn_implementation"] = attn_implementation

            eval_logger.info(f"Model kwargs: {model_kwargs}")

            try:
                self._model = AutoModel.from_pretrained(pretrained, **model_kwargs)
            except (AttributeError, ValueError) as e:
                if "_supports_sdpa" in str(e) or "attn_implementation" in str(e):
                    eval_logger.warning(f"Failed to load with attn_implementation={attn_implementation}: {e}")
                    eval_logger.warning("Retrying with attn_implementation='eager'")
                    model_kwargs["attn_implementation"] = "eager"
                    self._model = AutoModel.from_pretrained(pretrained, **model_kwargs)
                else:
                    raise
            self._model = self._model.eval()
            self._config = self._model.config

            if isinstance(final_device_map, str) and final_device_map in ["auto", "balanced", "balanced_low_0", "sequential"]:
                eval_logger.info("=" * 80)
                eval_logger.info("Model device distribution:")
                if hasattr(self._model, "hf_device_map"):
                    for name, device in self._model.hf_device_map.items():
                        eval_logger.info(f"  {name}: {device}")
                else:
                    eval_logger.warning("Model does not have hf_device_map attribute")
                eval_logger.info("=" * 80)

            eval_logger.info("ILLUME+ model loaded successfully")

        except Exception as e:
            eval_logger.error(f"Failed to load model: {e}")
            import traceback
            eval_logger.error(traceback.format_exc())
            raise

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
        else:
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
        new_list = []
        for i in input_list:
            for j in i:
                new_list.append(j)
        return new_list

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood not implemented for ILLUME+")

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(
            total=len(requests),
            disable=(self.rank != 0),
            desc="Model Responding",
        )

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
            visuals = [
                doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id
            ]
            visuals = self.flatten(visuals)

            gen_kwargs = all_gen_kwargs[0]
            if "until" in gen_kwargs:
                gen_kwargs.pop("until")

            if isinstance(contexts, tuple):
                contexts = list(contexts)

            assert len(contexts) == 1, "Batch size must be 1"
            context = contexts[0]

            # Prepare images
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

            eval_logger.debug(f"Processed {len(images)} images from {len(visuals)} visuals")

            # Resize images to consistent dimensions
            if len(images) > 1:
                # Find the maximum dimensions across all images
                max_width = max(img.width for img in images)
                max_height = max(img.height for img in images)

                # =================================================================
                # [MODIFIED] Dynamic Resolution Strategy for OOM Prevention
                # =================================================================
                # 如果是多图（如 MMSI），强制降级到 512 以防止显存 OOM
                # 如果是单图，保持 1024 以获得最佳细节
                if len(images) > 1:
                    max_dim = 512
                    # eval_logger.debug(f"Multi-image detected ({len(images)}), clamping max_dim to {max_dim}")
                else:
                    max_dim = 1024
                # =================================================================

                if max_width > max_dim or max_height > max_dim:
                    scale = max_dim / max(max_width, max_height)
                    max_width = int(max_width * scale)
                    max_height = int(max_height * scale)

                # Resize all images to the same dimensions
                resized_images = []
                for img in images:
                    if img.size != (max_width, max_height):
                        img = img.resize((max_width, max_height), Image.Resampling.LANCZOS)
                    resized_images.append(img)
                images = resized_images
                eval_logger.debug(f"Resized {len(images)} images to {max_width}x{max_height}")

            # Set generation parameters
            max_new_tokens = gen_kwargs.get("max_new_tokens", self.max_new_tokens)
            temperature = gen_kwargs.get("temperature", 0.0)
            top_p = gen_kwargs.get("top_p", None)
            num_beams = gen_kwargs.get("num_beams", 1)

            try:
                response = self._generate_response(
                    context=context,
                    images=images,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    num_beams=num_beams,
                )
            except Exception as e:
                eval_logger.error(f"Generation error: {e}")
                import traceback
                eval_logger.error(traceback.format_exc())
                response = ""

            torch.cuda.empty_cache()

            res.append(response)
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), response)
            pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        return res

    def _generate_response(
        self,
        context: str,
        images: List[Image.Image],
        max_new_tokens: int,
        temperature: float,
        top_p: Optional[float],
        num_beams: int,
    ) -> str:
        # Handle case where no images are provided
        if not images:
            eval_logger.warning("No images provided for generation, returning empty response")
            return ""

        import re
        image_placeholder_patterns = [
            r'<image>', r'<img>', r'\[image\]', r'<IMAGE>', r'<IMG>',
        ]

        context_has_placeholders = any(
            re.search(pattern, context) for pattern in image_placeholder_patterns
        )

        if context_has_placeholders:
            eval_logger.debug("Context contains image placeholders, using context as-is")
            conversation = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a helpful assistant."}],
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": context}],
                },
            ]
        else:
            conversation = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a helpful assistant."}],
                },
                {
                    "role": "user",
                    "content": [{"type": "image"}] * len(images)
                    + [{"type": "text", "text": context}],
                },
            ]

        inputs = self._processor(text=conversation, images=images, return_tensors="pt")

        if self._use_device_map:
            target_device = "cuda:0" if torch.cuda.is_available() else "cuda"
            inputs = inputs.to(target_device)
        else:
            inputs = inputs.to(self._device)

        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "use_cache": self.use_cache,
        }

        if temperature > 0:
            generate_kwargs["do_sample"] = True
            generate_kwargs["temperature"] = temperature
            if top_p is not None:
                generate_kwargs["top_p"] = top_p
        else:
            generate_kwargs["do_sample"] = False

        if num_beams > 1:
            generate_kwargs["num_beams"] = num_beams

        with torch.no_grad():
            outputs = self._model.generate(**inputs, **generate_kwargs)

        outputs = outputs[:, inputs["input_ids"].shape[1] :]
        answer = self._processor.batch_decode(outputs, skip_special_tokens=True)[0]

        del outputs, inputs
        return answer

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("Multi-round generation not yet implemented")