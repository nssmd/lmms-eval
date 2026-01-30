"""
ILLUME+ Visual Chain-of-Thought Model

Two-stage inference:
1. Stage 1: Generate visualization image from text prompt using ILLUME+ generation
2. Stage 2: Answer question using both original and generated images

Usage:
    python -m lmms_eval \
        --model illume_plus_visual_cot \
        --model_args pretrained=ILLUME-MLLM/illume_plus-qwen2_5-7b-hf \
        --tasks mme \
        --batch_size 1 \
        --device cuda:0
"""

# CRITICAL: Disable flash_attn BEFORE any other imports
# This must be done at module load time to prevent incompatible flash_attn from being loaded
import os
os.environ['DIFFUSERS_DISABLE_FLASH_ATTN'] = '1'

import json
import re
import sys
from io import BytesIO
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import LogitsProcessorList

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model, MODEL_REGISTRY

# Prevent duplicate registration
if "illume_plus_visual_cot" in MODEL_REGISTRY:
    eval_logger.warning(
        "illume_plus_visual_cot already registered, skipping re-registration"
    )
    del MODEL_REGISTRY["illume_plus_visual_cot"]


@register_model("illume_plus_visual_cot")
class ILLUMEPlusVisualCoT(lmms):
    """
    ILLUME+ Visual Chain-of-Thought Model

    Performs two-stage visual reasoning:
    1. Generate visualization image from text prompt
    2. Answer question using both original and generated images
    """

    def __init__(
        self,
        pretrained: str = "ILLUME-MLLM/illume_plus-qwen2_5-7b-hf",
        device: str = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "bfloat16",
        batch_size: int = 1,
        trust_remote_code: Optional[bool] = True,
        use_cache: bool = True,
        attn_implementation: Optional[str] = "sdpa",
        device_map: Optional[str] = None,
        infer_auto_device_map: bool = False,
        # Stage 1: Image generation parameters
        stage1_max_new_tokens: int = 4096,
        stage1_temperature: float = 0.7,
        stage1_top_p: Optional[float] = 0.9,
        stage1_num_beams: int = 1,
        # Stage 2: Visual understanding parameters
        stage2_max_new_tokens: int = 4096,  # Reduced from 4096 to save memory
        stage2_temperature: float = 0.0,
        stage2_top_p: Optional[float] = None,
        stage2_num_beams: int = 1,
        # Generation prompt template
        generation_prompt_template: str = (
            "Generate a detailed visual diagram or illustration to help answer "
            "this question: {question}"
        ),
        # Output and debugging
        output_dir: Optional[str] = None,
        save_intermediate: bool = False,
        intermediate_dir: Optional[str] = None,
        # Error handling
        fail_gracefully: bool = True,
        # Vision tokenizer/decoder parameters
        enable_image_decoding: bool = True,
        tokenizer_config_path: Optional[str] = None,
        tokenizer_checkpoint: Optional[str] = None,
        diffusion_decoder_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        # Debug: Log received parameters
        eval_logger.info(f"Initializing with tokenizer_config_path={tokenizer_config_path}")
        eval_logger.info(f"Initializing with diffusion_decoder_path={diffusion_decoder_path}")
        eval_logger.info(f"Initializing with enable_image_decoding={enable_image_decoding}")

        self.pretrained = pretrained
        self.use_cache = use_cache
        self.trust_remote_code = trust_remote_code
        self.save_intermediate = save_intermediate
        self.fail_gracefully = fail_gracefully
        self.generation_prompt_template = generation_prompt_template
        self.infer_auto_device_map = infer_auto_device_map
        self.device_map = device_map
        self.enable_image_decoding = enable_image_decoding
        self.tokenizer_config_path = tokenizer_config_path
        self.tokenizer_checkpoint = tokenizer_checkpoint
        self.diffusion_decoder_path = diffusion_decoder_path

        # Stage 1 parameters
        self.stage1_max_new_tokens = stage1_max_new_tokens
        self.stage1_temperature = stage1_temperature
        self.stage1_top_p = stage1_top_p
        self.stage1_num_beams = stage1_num_beams

        # Stage 2 parameters
        self.stage2_max_new_tokens = stage2_max_new_tokens
        self.stage2_temperature = stage2_temperature
        self.stage2_top_p = stage2_top_p
        self.stage2_num_beams = stage2_num_beams

        # Setup output directories
        if output_dir is None:
            self.output_dir = "./logs/illume_plus_visual_cot"
        else:
            self.output_dir = output_dir

        self.generated_images_dir = os.path.join(self.output_dir, "generated_images")
        os.makedirs(self.generated_images_dir, exist_ok=True)

        if intermediate_dir is None:
            self.intermediate_dir = os.path.join(
                self.output_dir, "intermediate_artifacts"
            )
        else:
            self.intermediate_dir = intermediate_dir

        if save_intermediate:
            os.makedirs(self.intermediate_dir, exist_ok=True)
            eval_logger.info(
                f"Intermediate artifacts will be saved to: {self.intermediate_dir}"
            )

        # Setup accelerator for multi-GPU support
        eval_logger.info("Initializing Accelerator for multi-GPU support")
        try:
            accelerator = Accelerator()
            eval_logger.info(
                f"Accelerator initialized, num_processes = {accelerator.num_processes}"
            )
            if accelerator.num_processes > 1:
                self._device = torch.device(f"cuda:{accelerator.local_process_index}")
                self._use_accelerator = True
                self._accelerator = accelerator
            else:
                self._device = (
                    torch.device(device) if isinstance(device, str) else device
                )
                self._use_accelerator = False
                self._accelerator = None
        except Exception as e:
            eval_logger.warning(
                f"Accelerator initialization failed: {e}, using single device mode"
            )
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
        self._load_model(pretrained, attn_implementation)

        # Initialize ILLUME+ generation components (logits processor, special tokens)
        eval_logger.info("Initializing ILLUME+ generation components")
        self._init_illume_generation_components()

        # Load vision tokenizer/decoder if enabled
        self.vq_model = None
        self.diffusion_decoder_pipe = None
        if self.enable_image_decoding:
            eval_logger.info(
                "Image decoding is enabled, loading vision tokenizer/decoder"
            )
            self._load_vision_decoder()
        else:
            eval_logger.warning(
                "Image decoding is DISABLED. Generated images will be blank placeholders. "
                "To enable actual image generation, set enable_image_decoding=True and provide "
                "tokenizer_config_path, tokenizer_checkpoint, and diffusion_decoder_path."
            )

        self.batch_size_per_gpu = int(batch_size)
        assert self.batch_size_per_gpu == 1, "batch_size > 1 not supported"

        # Setup distributed training if using accelerator
        if self._use_accelerator and self._accelerator.num_processes > 1:
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

        eval_logger.info("ILLUME+ Visual CoT model initialized successfully")

    def _load_model(self, pretrained: str, attn_implementation: Optional[str]):
        """Load ILLUME+ model and processor."""
        try:
            from transformers import AutoModel, AutoProcessor
            import os
            import sys
            import time

            # Bypass torch.load security check for .bin files
            os.environ["TRANSFORMERS_ALLOW_UNSAFE_LOAD"] = "1"
            os.environ["TOKENIZERS_PARALLELISM"] = "false"

            eval_logger.info("Loading ILLUME+ model with transformers")

            # Check if model path exists
            if os.path.exists(pretrained):
                eval_logger.info(f"Loading from local path: {pretrained}")
                try:
                    files = os.listdir(pretrained)
                    eval_logger.info(f"Found {len(files)} files in model directory")

                    # Check for weight files
                    weight_files = [
                        f
                        for f in files
                        if f.endswith((".safetensors", ".bin")) and "pytorch_model" in f
                    ]
                    index_files = [f for f in files if f.endswith(".index.json")]

                    eval_logger.info(f"Weight files: {weight_files}")
                    eval_logger.info(f"Index files: {index_files}")

                    # If index file exists but no weight files, model is incomplete
                    if index_files and not weight_files:
                        raise ValueError(
                            f"Model directory {pretrained} contains index file but "
                            f"no weight files!\nPlease download the complete model "
                            f"weights or use HuggingFace Hub: "
                            f"pretrained=ILLUME-MLLM/illume_plus-qwen2_5-7b-hf"
                        )
                except Exception as e:
                    if "index file but no weight files" in str(e):
                        raise
                    eval_logger.warning(f"Could not list directory: {e}")
            else:
                eval_logger.info(f"Loading from HuggingFace Hub: {pretrained}")

            # Load processor
            eval_logger.info(f"Loading processor from {pretrained}")
            processor_kwargs = {
                "trust_remote_code": self.trust_remote_code,
            }

            # Disable flash_attn if it's causing import errors
            # This is a workaround for version incompatibility issues
            flash_attn_backup = sys.modules.get('flash_attn', None)
            if 'flash_attn' in sys.modules:
                eval_logger.warning(
                    "Temporarily disabling flash_attn to avoid import errors. "
                    "Model will use standard attention."
                )
                del sys.modules['flash_attn']

            # Also set environment variable to disable flash_attn in diffusers
            old_disable_flash = os.environ.get('DIFFUSERS_DISABLE_FLASH_ATTN')
            os.environ['DIFFUSERS_DISABLE_FLASH_ATTN'] = '1'

            start_time = time.time()
            try:
                self._processor = AutoProcessor.from_pretrained(
                    pretrained, **processor_kwargs
                )
            finally:
                # Restore flash_attn if it was there
                if flash_attn_backup is not None:
                    sys.modules['flash_attn'] = flash_attn_backup
                # Restore environment variable
                if old_disable_flash is None:
                    os.environ.pop('DIFFUSERS_DISABLE_FLASH_ATTN', None)
                else:
                    os.environ['DIFFUSERS_DISABLE_FLASH_ATTN'] = old_disable_flash
            elapsed = time.time() - start_time
            eval_logger.info(f"Processor loaded in {elapsed:.1f} seconds")

            self._tokenizer = self._processor.tokenizer
            eval_logger.info("Processor loaded successfully")

            # Load model with proper device handling
            eval_logger.info(f"Loading model from {pretrained} to {self._device}")

            # Determine device_map strategy
            if self.infer_auto_device_map:
                final_device_map = "auto"
                eval_logger.info(
                    "Using infer_auto_device_map for multi-GPU model parallelism"
                )
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

            # Try with specified attn_implementation first, fallback to eager if it fails
            if attn_implementation is not None:
                model_kwargs["attn_implementation"] = attn_implementation

            eval_logger.info(f"Model kwargs: {model_kwargs}")

            try:
                self._model = AutoModel.from_pretrained(pretrained, **model_kwargs)
            except (AttributeError, ValueError) as e:
                if "_supports_sdpa" in str(e) or "attn_implementation" in str(e):
                    eval_logger.warning(
                        f"Failed to load with attn_implementation={attn_implementation}: {e}"
                    )
                    eval_logger.warning("Retrying with attn_implementation='eager'")
                    model_kwargs["attn_implementation"] = "eager"
                    self._model = AutoModel.from_pretrained(pretrained, **model_kwargs)
                else:
                    raise

            self._model = self._model.eval()
            self._config = self._model.config

            eval_logger.info("ILLUME+ model loaded successfully")

        except ImportError as e:
            raise ImportError(
                f"Failed to import transformers. Please install it:\n"
                f"  pip install transformers\n"
                f"Error: {e}"
            )
        except Exception as e:
            eval_logger.error(f"Failed to load model: {e}")
            import traceback

            eval_logger.error(traceback.format_exc())
            raise

    def _init_illume_generation_components(self):
        """Initialize ILLUME+ specific generation components."""
        try:
            # Try to import from ILLUME_plus directory first (preferred for full feature support)
            illume_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "ILLUME_plus",
                "ILLUME",
            )

            processor_loaded = False
            if os.path.exists(illume_path):
                if illume_path not in sys.path:
                    sys.path.insert(0, illume_path)
                    eval_logger.info(f"Added ILLUME path to sys.path: {illume_path}")

                try:
                    from generation_eval.models.inference_utils import (
                        InterleavedLogitsProcessor,
                    )

                    self.InterleavedLogitsProcessor = InterleavedLogitsProcessor
                    self._processor_supports_image_sizes = True
                    eval_logger.info(
                        "Successfully imported InterleavedLogitsProcessor from ILLUME_plus directory (full feature support)"
                    )
                    processor_loaded = True
                except ImportError as e:
                    eval_logger.warning(
                        f"Failed to import from ILLUME_plus directory: {e}"
                    )

            # Fallback to model directory if ILLUME_plus import failed
            if not processor_loaded:
                model_path = self.pretrained
                if os.path.exists(model_path) and model_path not in sys.path:
                    sys.path.insert(0, model_path)
                    eval_logger.info(f"Added model path to sys.path: {model_path}")

                from inference_utils import InterleavedLogitsProcessor

                self.InterleavedLogitsProcessor = InterleavedLogitsProcessor
                self._processor_supports_image_sizes = False
                eval_logger.info(
                    "Successfully imported InterleavedLogitsProcessor from model directory (limited feature support)"
                )

            eval_logger.info(
                f"InterleavedLogitsProcessor loaded successfully (image_sizes support: {self._processor_supports_image_sizes})"
            )

            # Define special tokens for Qwen2.5
            self.special_tokens_ids = [
                151665,
                151666,
                151667,
                151668,
                151669,
                151670,
                151671,
            ]
            start_token = 151672 + 32
            self.level0_range = (start_token, start_token + 32768)
            self.level1_range = (start_token + 32768, start_token + 32768 * 4)

            self.special_tokens_dict = {
                "start_of_image": 151665,
                "end_of_image": 151666,
                "start_of_level0": 151668,
                "end_of_level0": 151669,
                "start_of_level1": 151670,
                "end_of_level1": 151671,
                "end_of_line": 151667,
                "end_of_text": 151645,
                "level0_range": self.level0_range,
                "level1_range": self.level1_range,
            }

            eval_logger.info("ILLUME+ generation components initialized successfully")

        except ImportError as e:
            eval_logger.error(
                f"Failed to import ILLUME+ generation utilities: {e}. "
                f"Image generation will not work properly."
            )
            self.InterleavedLogitsProcessor = None
            self.special_tokens_dict = None
            self._processor_supports_image_sizes = False
        except Exception as e:
            eval_logger.error(f"Failed to initialize ILLUME+ components: {e}")
            import traceback

            eval_logger.error(traceback.format_exc())
            self.InterleavedLogitsProcessor = None
            self.special_tokens_dict = None
            self._processor_supports_image_sizes = False

    def _calculate_image_token_dimensions(
        self, h: int, w: int, downsample_rate_per_level: List[int] = [28, 16]
    ) -> Tuple[int, int, int, int, int]:
        """
        Calculate image token dimensions for given resolution.

        Args:
            h: Image height
            w: Image width
            downsample_rate_per_level: Downsampling rates for each level

        Returns:
            Tuple of (semantic_token_num, pixel_token_num, h1, w1, h2, w2)
        """
        # Try to import RESOLUTION_MAPPING
        try:
            # Ensure vision_tokenizer is in path
            import sys
            import os

            project_root = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            )
            vision_tokenizer_path = os.path.join(
                project_root, "ILLUME_plus", "vision_tokenizer"
            )
            if (
                os.path.exists(vision_tokenizer_path)
                and vision_tokenizer_path not in sys.path
            ):
                sys.path.insert(0, vision_tokenizer_path)

            from tokenizer.dualvitok_model import RESOLUTION_MAPPING

            mapped_w, mapped_h = RESOLUTION_MAPPING.get((w, h), (w, h))
        except ImportError:
            eval_logger.warning(
                "Could not import RESOLUTION_MAPPING, using original resolution"
            )
            mapped_w, mapped_h = w, h

        # Level 0 - Semantic tokens
        w1 = mapped_w // downsample_rate_per_level[0]
        h1 = mapped_h // downsample_rate_per_level[0]
        semantic_token_num = w1 * h1

        # Level 1 - Pixel tokens
        w2 = w // downsample_rate_per_level[1]
        h2 = h // downsample_rate_per_level[1]
        pixel_token_num = w2 * h2

        return semantic_token_num, pixel_token_num, h1, w1, h2, w2

    def _load_vision_decoder(self):
        try:
            import os
            import sys
            from types import ModuleType
            import importlib.machinery
            from transformers import AutoModel
            import transformers.utils.import_utils as import_utils

            import_utils.is_flash_attn_2_available = lambda: False

            if not self.tokenizer_config_path:
                eval_logger.error("tokenizer_config_path is required for image decoding")
                return

            eval_logger.info("Loading vision tokenizer via Official HF style...")
            
            model_dir = self.tokenizer_config_path
            if os.path.isfile(self.tokenizer_config_path):
                model_dir = os.path.dirname(self.tokenizer_config_path)

            eval_logger.info(f"Targeting model directory: {model_dir}")

            flash_attn_available = False
            try:
                import flash_attn
                flash_attn_available = True
            except (ImportError, RuntimeError, ValueError):
                eval_logger.warning("flash_attn not available or broken, creating a robust mock.")
                
                mock_name = 'flash_attn'
                mock_module = ModuleType(mock_name)
                mock_module.__spec__ = importlib.machinery.ModuleSpec(mock_name, None)
                sys.modules[mock_name] = mock_module

            try:
                dualvitok = (
                    AutoModel.from_pretrained(
                        model_dir, 
                        trust_remote_code=True, 
                        torch_dtype=self._dtype,
                        attn_implementation="sdpa" 
                    )
                    .to(self._device)
                    .eval()
                )
            finally:
                if not flash_attn_available and 'flash_attn' in sys.modules:
                    del sys.modules['flash_attn']

            if hasattr(self._processor, "set_vision_tokenizer"):
                self._processor.set_vision_tokenizer(dualvitok)
                eval_logger.info("Vision tokenizer (DualViTok) linked to processor.")
            else:
                eval_logger.warning("Processor does not have set_vision_tokenizer method.")

            self.vq_model = dualvitok

            if self.diffusion_decoder_path:
                eval_logger.info(f"Loading SDXL via processor: {self.diffusion_decoder_path}")
                if hasattr(self._processor, "load_diffusion_vision_detokenizer"):
                    self._processor.load_diffusion_vision_detokenizer(
                        self.diffusion_decoder_path
                    )
                    self.diffusion_decoder_pipe = getattr(
                        self._processor, "diffusion_model", None
                    )
                    eval_logger.info("Diffusion decoder loaded successfully.")
                else:
                    eval_logger.warning("Processor does not support load_diffusion_vision_detokenizer.")

            eval_logger.info("Vision decoder initialization complete via official HF style.")

        except Exception as e:
            eval_logger.error(f"Failed to load via official style: {e}")
            import traceback
            eval_logger.error(traceback.format_exc())

    def _extract_image_tokens_from_text(
        self, text: str, num_levels: int = 2
    ) -> Optional[List[List[int]]]:
        """
        Extract image tokens from generated text.

        Args:
            text: Generated text containing image tokens
            num_levels: Number of token levels (default: 2 for semantic + pixel)

        Returns:
            List of token lists for each level, or None if no tokens found
        """
        try:
            image_embed_inds = []
            for level in range(num_levels):
                pattern = r"<\|image_level{}_(\d+)\|>".format(level)
                matches = re.findall(pattern, text)
                image_embed_ind = [int(num) for num in matches]
                image_embed_inds.append(image_embed_ind)

            # Check if we found any tokens
            if all(len(tokens) == 0 for tokens in image_embed_inds):
                return None

            return image_embed_inds

        except Exception as e:
            eval_logger.error(f"Failed to extract image tokens: {e}")
            return None

    def _decode_image_tokens(
        self,
        image_tokens: List[List[int]],
        resolution: Tuple[int, int],
        use_diffusion: bool = True,
    ) -> Optional[np.ndarray]:
        """
        Decode image tokens to actual image.

        Args:
            image_tokens: List of [semantic_tokens, pixel_tokens]
            resolution: Target resolution (h, w)
            use_diffusion: Whether to use diffusion decoder

        Returns:
            Decoded image as numpy array, or None if decoding fails
        """
        try:
            if self.vq_model is None:
                eval_logger.warning("VQ model not loaded, cannot decode image")
                return None

            if len(image_tokens) < 2:
                eval_logger.warning(f"Expected 2 token levels, got {len(image_tokens)}")
                return None

            semantic_tokens = image_tokens[0]
            pixel_tokens = image_tokens[1]

            if len(semantic_tokens) == 0 or len(pixel_tokens) == 0:
                eval_logger.warning("Empty token lists, cannot decode image")
                return None

            # Calculate expected dimensions
            h, w = resolution

            # Ensure vision_tokenizer is in path
            import sys
            import os

            project_root = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            )
            vision_tokenizer_path = os.path.join(
                project_root, "ILLUME_plus", "vision_tokenizer"
            )
            if (
                os.path.exists(vision_tokenizer_path)
                and vision_tokenizer_path not in sys.path
            ):
                sys.path.insert(0, vision_tokenizer_path)

            try:
                from tokenizer.dualvitok_model import RESOLUTION_MAPPING

                mapped_w, mapped_h = RESOLUTION_MAPPING.get(
                    (w, h), (w, h)
                )  # Fallback to original if not in mapping
            except ImportError:
                eval_logger.warning(
                    "Could not import RESOLUTION_MAPPING, using original resolution"
                )
                mapped_w, mapped_h = w, h

            # Semantic tokens: downsampled by 28
            h1 = mapped_h // 28
            w1 = mapped_w // 28
            expected_semantic = h1 * w1

            # Pixel tokens: downsampled by 16
            h2 = h // 16
            w2 = w // 16
            expected_pixel = h2 * w2

            eval_logger.debug(
                f"Expected tokens - semantic: {expected_semantic} ({h1}x{w1}), "
                f"pixel: {expected_pixel} ({h2}x{w2})"
            )
            eval_logger.debug(
                f"Got tokens - semantic: {len(semantic_tokens)}, "
                f"pixel: {len(pixel_tokens)}"
            )

            # Convert to tensors and reshape
            semantic_code = torch.as_tensor([semantic_tokens])
            pixel_code = torch.as_tensor([pixel_tokens])

            # Reshape to 2D grids
            try:
                semantic_code = semantic_code.view(1, h1, w1)
                pixel_code = pixel_code.view(1, h2, w2)
            except RuntimeError as e:
                eval_logger.error(
                    f"Failed to reshape tokens: {e}. "
                    f"Semantic: {len(semantic_tokens)} -> (1, {h1}, {w1}), "
                    f"Pixel: {len(pixel_tokens)} -> (1, {h2}, {w2})"
                )
                return None

            # Decode using diffusion decoder if available
            if use_diffusion and self.diffusion_decoder_pipe is not None:
                eval_logger.debug("Using diffusion decoder")
                diffusion_outputs = self.diffusion_decoder_pipe(
                    vq_indices=(semantic_code, pixel_code),
                    height=h * 2,
                    width=w * 2,
                    guidance_scale=1.5,
                    num_inference_steps=50,
                    generator=torch.Generator(self._device).manual_seed(42),
                )
                samples = diffusion_outputs.images
                decoded_image = np.asarray(samples[0])
            else:
                # Use VQ decoder
                eval_logger.debug("Using VQ decoder")
                quant_semantic = self.vq_model.semantic_quantizer.indices_to_codes(
                    semantic_code
                )
                quant_pixel = self.vq_model.pixel_quantizer.indices_to_codes(pixel_code)
                samples = self.vq_model.decode(quant_semantic, quant_pixel)
                decoded_image = (
                    torch.clamp(127.5 * samples + 128.0, 0, 255)
                    .permute(0, 2, 3, 1)
                    .to("cpu", dtype=torch.uint8)
                    .numpy()[0]
                )

            return decoded_image

        except Exception as e:
            eval_logger.error(f"Failed to decode image tokens: {e}")
            import traceback

            eval_logger.error(traceback.format_exc())
            return None

    @property
    def config(self):
        """Return the model config."""
        return self._config

    @property
    def tokenizer(self):
        """Return the tokenizer."""
        return self._tokenizer

    @property
    def processor(self):
        """Return the processor."""
        return self._processor

    @property
    def model(self):
        """Return the model, unwrapping it if using Accelerate."""
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        """Return the end of text token id."""
        return self.tokenizer.eos_token_id

    @property
    def batch_size(self):
        """Return the batch size."""
        return self.batch_size_per_gpu

    @property
    def device(self):
        """Return the device."""
        return self._device

    @property
    def rank(self):
        """Return the process rank."""
        return self._rank

    @property
    def world_size(self):
        """Return the world size."""
        return self._world_size

    def flatten(self, input_list):
        """Flatten a nested list."""
        new_list = []
        for i in input_list:
            for j in i:
                new_list.append(j)
        return new_list

    def _extract_image_from_various_formats(self, img_data) -> Optional[Image.Image]:
        """Extract PIL Image from various formats."""
        try:
            if img_data is None:
                return None
            elif isinstance(img_data, Image.Image):
                return img_data.convert("RGB")
            elif isinstance(img_data, str):
                return Image.open(img_data).convert("RGB")
            elif isinstance(img_data, dict):
                if "bytes" in img_data:
                    return Image.open(BytesIO(img_data["bytes"])).convert("RGB")
                elif "path" in img_data:
                    return Image.open(img_data["path"]).convert("RGB")
                elif "image" in img_data:
                    return self._extract_image_from_various_formats(img_data["image"])
            elif hasattr(img_data, "convert"):
                return img_data.convert("RGB")
            else:
                return Image.open(img_data).convert("RGB")
        except Exception as e:
            eval_logger.debug(f"Failed to extract image: {e}")
            return None

    def _normalize_image_sizes(self, images: List[Image.Image]) -> List[Image.Image]:
        """
        Normalize all images to have consistent dimensions.

        This ensures that the image processor can create homogeneous arrays.
        Uses the size of the first image as the target size.
        """
        if not images:
            return images

        # Get target size from first image
        target_size = images[0].size

        normalized_images = []
        for img in images:
            if img.size != target_size:
                eval_logger.debug(f"Resizing image from {img.size} to {target_size}")
                img = img.resize(target_size, Image.Resampling.LANCZOS)
            normalized_images.append(img)

        return normalized_images

    def _stage1_generate_image(
        self, generation_prompt: str, doc_id: str, task: str, original_images=None
    ) -> Tuple[str, List[str]]:
        try:
            images = []
            if original_images is not None:
                if not isinstance(original_images, list):
                    original_images = [original_images]
                for img in original_images:
                    extracted_img = self._extract_image_from_various_formats(img)
                    if extracted_img is not None:
                        images.append(extracted_img)

            if images:
                h, w = images[0].size[1], images[0].size[0]
            else:
                h, w = 512, 512

            # ILLUME+ only supports specific resolutions
            SUPPORTED_RESOLUTIONS = [
                (256, 256), (512, 512), (384, 640), (640, 384),
                (512, 384), (384, 512), (256, 384), (384, 256),
                (256, 512), (512, 256)
            ]

            # Find the closest supported resolution
            def find_closest_resolution(target_h, target_w, supported_resolutions):
                """Find the closest supported resolution that maintains aspect ratio."""
                target_ratio = target_w / target_h
                best_resolution = None
                best_score = float('inf')

                for res_h, res_w in supported_resolutions:
                    res_ratio = res_w / res_h
                    # Score based on aspect ratio difference and size difference
                    ratio_diff = abs(res_ratio - target_ratio)
                    size_diff = abs(res_h * res_w - target_h * target_w) / (target_h * target_w)
                    score = ratio_diff + 0.1 * size_diff

                    if score < best_score:
                        best_score = score
                        best_resolution = (res_h, res_w)

                return best_resolution

            # Use closest supported resolution
            h, w = find_closest_resolution(h, w, SUPPORTED_RESOLUTIONS)
            eval_logger.info(f"Using supported resolution: {h}x{w}")

            resolution_tag = f"<height_{h}><width_{w}>"

            # CRITICAL: Force image generation by explicitly requesting it in the prompt
            # Similar to MIO's approach, we make it clear that an image MUST be generated
            if images:
                # Image editing mode - explicitly request edited image output
                full_prompt = (
                    f"{resolution_tag}\n"
                    f"Edit the image according to this instruction: {generation_prompt}"
                )
                uncond_prompt = f"{resolution_tag}\nReconstruct the image according to the given image\n"
            else:
                # Image generation mode - explicitly request image output
                full_prompt = (
                    f"{resolution_tag}\n"
                    f"Generate an image with the following content: {generation_prompt}"
                )
                uncond_prompt = f"Generate a random image of {resolution_tag}\n"

            eval_logger.info(f"Generation prompt: {full_prompt}")
            eval_logger.info(f"Unconditional prompt: {uncond_prompt[:200]}")

            if images:
                if len(images) > 1:
                    images = self._normalize_image_sizes(images)
                conversation = [
                    {
                        "role": "system",
                        "content": [
                            {"type": "text", "text": "You are a helpful assistant."}
                        ],
                    },
                    {
                        "role": "user",
                        "content": [{"type": "image"}] * len(images)
                        + [{"type": "text", "text": full_prompt}],
                    },
                ]
                inputs = self._processor(
                    text=conversation, images=images, return_tensors="pt"
                )
            else:
                conversation = [
                    {
                        "role": "system",
                        "content": [
                            {"type": "text", "text": "You are a helpful assistant."}
                        ],
                    },
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": full_prompt}],
                    },
                ]
                inputs = self._processor(
                    text=conversation, images=None, return_tensors="pt"
                )

            inputs = inputs.to(self._device)

            # Process unconditional prompt for classifier-free guidance
            if images:
                # For image editing with CFG
                uncond_conversation = [
                    {
                        "role": "system",
                        "content": [
                            {"type": "text", "text": "You are a helpful assistant."}
                        ],
                    },
                    {
                        "role": "user",
                        "content": [{"type": "image"}] * len(images)
                        + [{"type": "text", "text": uncond_prompt}],
                    },
                ]
                uncond_inputs = self._processor(
                    text=uncond_conversation, images=images, return_tensors="pt"
                )
                uncond_inputs = uncond_inputs.to(self._device)
            else:
                # For image generation with CFG
                uncond_conversation = [
                    {
                        "role": "system",
                        "content": [
                            {"type": "text", "text": "You are a helpful assistant."}
                        ],
                    },
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": uncond_prompt}],
                    },
                ]
                uncond_inputs = self._processor(
                    text=uncond_conversation, images=None, return_tensors="pt"
                )
                uncond_inputs = uncond_inputs.to(self._device)

            semantic_token_num, pixel_token_num, h1, w1, h2, w2 = (
                self._calculate_image_token_dimensions(h, w)
            )
            # CRITICAL: Calculate expected tokens and add generous buffer
            # Format: <start_of_image> + semantic_tokens + <end_of_level0> + pixel_tokens + <end_of_image>
            expected_image_tokens = (h1 * (w1 + 1) + 2) + (h2 * (w2 + 1) + 2) + 50
            
            eval_logger.info(f"Expected image tokens: {expected_image_tokens} (semantic: {semantic_token_num}, pixel: {pixel_token_num})")
            eval_logger.info(f"Token dimensions: semantic={h1}x{w1}, pixel={h2}x{w2}")

            # Use different parameters for image editing vs generation
            if images:
                # Image editing parameters - ensure enough tokens for full image
                generate_kwargs = {
                    "max_new_tokens": max(8192, expected_image_tokens + 1000),  # Generous buffer
                    "use_cache": self.use_cache,
                    "do_sample": True,
                    "temperature": 1.0,
                    "top_p": 1.0,
                }
                # Use moderate sampling to ensure quality
                default_temp = 1.0
                level0_temp = 1.0
                level1_temp = 1.0
                level0_top_k = 2048
                level1_top_k = 2048 * 3
                guidance_scale = 1.0  # Start with no CFG for stability
                
                eval_logger.info("Using image EDITING mode with moderate sampling")
            else:
                # Image generation parameters - ensure enough tokens for full image
                generate_kwargs = {
                    "max_new_tokens": max(8192, expected_image_tokens + 1000),  # Generous buffer
                    "use_cache": self.use_cache,
                    "do_sample": True,
                    "temperature": 0.7,
                    "top_p": 0.9,
                }
                default_temp = 0.7
                level0_temp = 0.7
                level1_temp = 0.7
                level0_top_k = 2048
                level1_top_k = 6144
                guidance_scale = 1.0
                
                eval_logger.info("Using image GENERATION mode with standard sampling")

            if self.InterleavedLogitsProcessor is None:
                raise RuntimeError(
                    "InterleavedLogitsProcessor is not available. "
                    "Image generation requires the ILLUME+ generation utilities. "
                    "Please ensure the ILLUME_plus directory is properly set up."
                )

            try:
                processor_kwargs = {
                    "guidance_scale": guidance_scale,
                    "uncond": uncond_inputs["input_ids"] if guidance_scale > 1.0 else None,
                    "attention_mask": uncond_inputs.get("attention_mask", None) if guidance_scale > 1.0 else None,
                    "model": self._model,
                    "level0_range": self.level0_range,
                    "level1_range": self.level1_range,
                    "num_level0_rows": h1,
                    "num_level0_tokens": w1,
                    "num_level1_rows": h2,
                    "num_level1_tokens": w2,
                    "special_tokens": self.special_tokens_dict,
                    "default_temp": default_temp,
                    "level0_temp": level0_temp,
                    "level1_temp": level1_temp,
                    "default_top_k": 2048,
                    "level0_top_k": level0_top_k,
                    "level1_top_k": level1_top_k,
                    "default_top_p": 0.9,
                    "images": inputs.get("pixel_values", None),
                }

                # Only add image_sizes if the processor supports it
                if self._processor_supports_image_sizes:
                    processor_kwargs["image_sizes"] = [(w, h)]

                logits_processor = self.InterleavedLogitsProcessor(**processor_kwargs)
                generate_kwargs["logits_processor"] = LogitsProcessorList(
                    [logits_processor]
                )
                mode = "image editing" if images else "image generation"
                eval_logger.info(
                    f"Successfully created InterleavedLogitsProcessor for {mode} ({h}x{w})"
                )
            except Exception as e:
                eval_logger.error(f"Failed to create logits processor: {e}")
                import traceback

                eval_logger.error(traceback.format_exc())
                raise RuntimeError(
                    f"Cannot generate images without logits processor: {e}"
                )

            # Add ILLUME+ specific image generation parameters
            image_gen_kwargs = {
                "target_image_resolution": (h, w),
                "guidance_scale": guidance_scale,
                "image_semantic_temperature": default_temp,
                "image_semantic_top_k": level0_top_k,
                "image_semantic_top_p": generate_kwargs.get("top_p", 1.0),
                "image_pixel_temperature": level1_temp,
                "image_pixel_top_k": level1_top_k,
                "image_pixel_top_p": generate_kwargs.get("top_p", 1.0),
            }

            # Add unconditional prompt for CFG if guidance_scale > 1
            if guidance_scale > 1.0:
                image_gen_kwargs["negative_image_prompt_ids"] = uncond_inputs["input_ids"]
                image_gen_kwargs["negative_image_prompt_attention_mask"] = uncond_inputs.get("attention_mask", None)

            # CRITICAL: Pre-fill <start_of_image> token to FORCE image generation
            # Similar to MIO's approach of pre-filling <image> token
            start_of_image_id = self.special_tokens_dict["start_of_image"]
            eval_logger.info(f"Pre-filling <start_of_image> token (ID: {start_of_image_id}) to force image generation")
            
            # Append start_of_image token to input_ids
            input_ids = inputs["input_ids"]
            start_token_tensor = torch.tensor([[start_of_image_id]], dtype=input_ids.dtype, device=input_ids.device)
            input_ids_with_trigger = torch.cat([input_ids, start_token_tensor], dim=1)
            
            # Update attention_mask accordingly
            attention_mask = inputs["attention_mask"]
            attention_mask_extended = torch.cat([attention_mask, torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=attention_mask.device)], dim=1)
            
            # Update inputs dict
            inputs["input_ids"] = input_ids_with_trigger
            inputs["attention_mask"] = attention_mask_extended
            
            eval_logger.info(f"Input shape after pre-filling: {input_ids_with_trigger.shape}")

            with torch.no_grad():
                eval_logger.info(f"Starting generation with max_new_tokens={generate_kwargs['max_new_tokens']}")
                outputs = self._model.generate(**inputs, **generate_kwargs, **image_gen_kwargs)

            outputs_text = outputs[:, inputs["input_ids"].shape[1] :]
            generated_text = self._processor.batch_decode(
                outputs_text, skip_special_tokens=False
            )[0]
            
            # Log generation statistics
            eval_logger.info(f"Generated {outputs_text.shape[1]} tokens")
            eval_logger.info(f"Generated text preview (first 200 chars): {generated_text[:200]}")

            image_tokens = self._extract_image_tokens_from_text(generated_text)
            
            # CRITICAL: Validate that we actually got image tokens
            if image_tokens is None or len(image_tokens) < 2:
                eval_logger.error(f"❌ FAILED to generate image tokens!")
                eval_logger.error(f"Generated text: {generated_text[:500]}")
                eval_logger.error(f"Image tokens: {image_tokens}")
            else:
                eval_logger.info(f"✅ Successfully generated image tokens:")
                eval_logger.info(f"   - Semantic tokens (L0): {len(image_tokens[0])} (expected: ~{semantic_token_num})")
                eval_logger.info(f"   - Pixel tokens (L1): {len(image_tokens[1])} (expected: ~{pixel_token_num})")
                
                # Validate token counts
                if len(image_tokens[0]) < semantic_token_num * 0.8:
                    eval_logger.warning(f"⚠️ Semantic tokens count is low: {len(image_tokens[0])} < {semantic_token_num * 0.8}")
                if len(image_tokens[1]) < pixel_token_num * 0.8:
                    eval_logger.warning(f"⚠️ Pixel tokens count is low: {len(image_tokens[1])} < {pixel_token_num * 0.8}")

            task_dir = os.path.join(self.generated_images_dir, task)
            os.makedirs(task_dir, exist_ok=True)
            image_path = os.path.join(task_dir, f"{doc_id}_gen.png")

            if image_tokens and self.enable_image_decoding:
                eval_logger.info(
                    f"Decoding captured tokens: L0={len(image_tokens[0])}, L1={len(image_tokens[1])}"
                )
                decoded_image = self._decode_image_tokens(
                    image_tokens, resolution=(h, w), use_diffusion=True
                )
                if decoded_image is not None:
                    Image.fromarray(decoded_image).save(image_path)
                    eval_logger.info(f"✅ Successfully saved decoded image to: {image_path}")
                else:
                    eval_logger.warning(f"⚠️ Image decoding failed, saving gray placeholder")
                    Image.new("RGB", (h, w), color=(128, 128, 128)).save(image_path)
            else:
                if not image_tokens:
                    eval_logger.error(f"❌ No image tokens found in generated text!")
                    eval_logger.error(f"Generated text preview: {generated_text[:100]}")
                if not self.enable_image_decoding:
                    eval_logger.warning(f"⚠️ Image decoding is disabled")
                    
                eval_logger.warning(f"Saving light gray placeholder image")
                Image.new("RGB", (h, w), color=(200, 200, 200)).save(image_path)

            return generated_text, [image_path]

        except Exception as e:
            eval_logger.error(f"Stage 1 generation error: {e}")
            if self.fail_gracefully:
                return "", []
            raise

    def _stage2_answer_with_images(
        self, question: str, generated_image_paths: List[str], original_images: List[Image.Image] = None
    ) -> str:
        eval_logger.debug("Stage 2 - Answering question with multiple images")
        try:
            images = []

            if original_images:
                for img in original_images:
                    extracted_img = self._extract_image_from_various_formats(img)
                    if extracted_img:
                        images.append(extracted_img)

            for img_path in generated_image_paths:
                gen_image = Image.open(img_path).convert("RGB")
                images.append(gen_image)

            if not images:
                return ""

            images = self._normalize_image_sizes(images)

            conversation = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "You are a helpful assistant."}
                    ],
                },
                {
                    "role": "user",
                    "content": (
                        [{"type": "image"}] * len(images)
                        + [{"type": "text", "text": question}]
                    ),
                },
            ]

            inputs = self._processor(
                text=conversation, images=images, return_tensors="pt"
            )

            if self.infer_auto_device_map or self.device_map == "auto":
                inputs = inputs.to("cuda")
            else:
                inputs = inputs.to(self._device)

            generate_kwargs = {
                "max_new_tokens": self.stage2_max_new_tokens,
                "use_cache": self.use_cache,
            }

            if self.stage2_temperature > 0:
                generate_kwargs["do_sample"] = True
                generate_kwargs["temperature"] = self.stage2_temperature
                if self.stage2_top_p is not None:
                    generate_kwargs["top_p"] = self.stage2_top_p
            else:
                generate_kwargs["do_sample"] = False

            if self.stage2_num_beams > 1:
                generate_kwargs["num_beams"] = self.stage2_num_beams

            with torch.no_grad():
                outputs = self._model.generate(**inputs, **generate_kwargs)

            outputs_text = outputs[:, inputs["input_ids"].shape[1] :]
            answer = self._processor.batch_decode(
                outputs_text, skip_special_tokens=True
            )[0]

            del outputs, inputs
            torch.cuda.empty_cache()

            return answer

        except Exception as e:
            eval_logger.error(f"Stage 2 error: {e}")
            if self.fail_gracefully:
                return ""
            raise
            

    def _save_intermediate_artifacts(
        self,
        doc_id: str,
        task: str,
        generation_prompt: str,
        stage1_text: str,
        generated_images: List[str],
        question: str,
        stage2_answer: str,
    ) -> None:
        """Save intermediate artifacts for debugging."""
        if not self.save_intermediate:
            return

        artifact_dir = os.path.join(self.intermediate_dir, task)
        os.makedirs(artifact_dir, exist_ok=True)

        # Save metadata
        metadata = {
            "doc_id": doc_id,
            "task": task,
            "generation_prompt": generation_prompt,
            "stage1_text": stage1_text,
            "generated_images": generated_images,
            "question": question,
            "stage2_answer": stage2_answer,
        }

        metadata_path = os.path.join(artifact_dir, f"{doc_id}_metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        eval_logger.debug(f"Saved intermediate artifacts to: {metadata_path}")

    def generate_uni_mmmu_interleaved(
        self,
        input_images: List,
        prompt: str,
        doc_id: str,
        task: str,
        interleaved_config: dict,
        doc: dict = None,
    ) -> Tuple[str, List[str]]:
        """
        Uni-MMMU interleaved generation for ILLUME+ Visual CoT.

        This implements the exact generation flow from the original Uni-MMMU:
        - Jigsaw: gen_image(cand0) → gen_image(cand1) → gen_text(answer)
        - Maze/Sliding: [gen_text(plan) → gen_image(step)]×k → gen_text(answer)

        Args:
            input_images: List of input images
            prompt: Base prompt text
            doc_id: Document ID for file naming
            task: Task name for file naming
            interleaved_config: Configuration dict from yaml
            doc: Document data for dynamic num_images extraction

        Returns:
            Tuple of (final_text_answer, list_of_generated_image_paths)
        """
        import json as json_module

        task_type = interleaved_config.get("task_type", "jigsaw")

        # Get num_images dynamically from doc if available
        num_images = interleaved_config.get("num_images", 2)
        if doc is not None:
            if task_type == "maze":
                # Get step count from ground truth
                steps_str = doc.get("steps", "[]")
                steps = (
                    json_module.loads(steps_str)
                    if isinstance(steps_str, str)
                    else steps_str
                )
                if steps:
                    num_images = len(steps)
            elif task_type == "sliding":
                # Get step count from ground truth
                steps_str = doc.get("steps_words", "[]")
                steps = (
                    json_module.loads(steps_str)
                    if isinstance(steps_str, str)
                    else steps_str
                )
                if steps:
                    num_images = len(steps)

        # Extract original images from input_images
        original_images = []
        if input_images:
            for img in input_images:
                extracted_img = self._extract_image_from_various_formats(img)
                if extracted_img is not None:
                    original_images.append(extracted_img)

        # For backward compatibility, use first image as single original_image
        original_image = original_images[0] if original_images else None

        generated_images = []

        if task_type == "jigsaw":
            # Jigsaw: Generate 2 completed images then final answer
            # Image 1: Candidate 0 completion
            suffix1 = "Output ONLY a single image with Candidate 0 placed in the bottom-right cell. No text."
            gen_prompt1 = prompt + "\n\n" + suffix1

            _, img_paths_0 = self._stage1_generate_image(
                generation_prompt=gen_prompt1,
                doc_id=f"{doc_id}_cand0",
                task=task,
                original_images=original_images,
            )
            if img_paths_0:
                generated_images.extend(img_paths_0)
                eval_logger.info(f"Saved jigsaw image 0: {img_paths_0[0]}")

            # Image 2: Candidate 1 completion
            suffix2 = "Output ONLY a single image with Candidate 1 placed in the bottom-right cell. No text."
            gen_prompt2 = prompt + "\n\n" + suffix2

            _, img_paths_1 = self._stage1_generate_image(
                generation_prompt=gen_prompt2,
                doc_id=f"{doc_id}_cand1",
                task=task,
                original_images=original_images,
            )
            if img_paths_1:
                generated_images.extend(img_paths_1)
                eval_logger.info(f"Saved jigsaw image 1: {img_paths_1[0]}")

            # Final answer using stage 2 with all generated images
            final_suffix = (
                'Now output EXACTLY ONE <FINAL_ANSWER_JSON>{"choice": 0 or 1, "rationale": "≤30 words"}</FINAL_ANSWER_JSON>\n'
                "Do not output any additional images."
            )
            final_question = prompt + "\n\n" + final_suffix

            # Use custom stage 2 logic with multiple images
            if len(generated_images) >= 2:
                # Load both generated images
                gen_img0 = Image.open(generated_images[0]).convert("RGB")
                gen_img1 = Image.open(generated_images[1]).convert("RGB")

                # Build images list
                images = []
                if original_image:
                    images.append(original_image)
                images.extend([gen_img0, gen_img1])

                # Normalize image sizes to ensure consistent dimensions
                images = self._normalize_image_sizes(images)

                # Build conversation format
                conversation = [
                    {
                        "role": "system",
                        "content": [
                            {"type": "text", "text": "You are a helpful assistant."}
                        ],
                    },
                    {
                        "role": "user",
                        "content": (
                            [{"type": "image"}] * len(images)
                            + [{"type": "text", "text": final_question}]
                        ),
                    },
                ]

                # Process inputs
                inputs = self._processor(
                    text=conversation, images=images, return_tensors="pt"
                )

                # Move inputs to appropriate device
                if self.infer_auto_device_map or self.device_map == "auto":
                    inputs = inputs.to("cuda")
                else:
                    inputs = inputs.to(self._device)

                # Generate answer
                with torch.no_grad():
                    output_ids = self._model.generate(
                        **inputs,
                        max_new_tokens=self.stage2_max_new_tokens,
                        do_sample=False,
                    )

                # Decode output
                input_len = inputs["input_ids"].shape[1]
                generated_ids = output_ids[0][input_len:]
                final_text = self.tokenizer.decode(
                    generated_ids, skip_special_tokens=True
                )

                # Clean up
                del inputs, output_ids, generated_ids
                torch.cuda.empty_cache()
            else:
                final_text = ""

        else:
            # Maze/Sliding: [gen_text(plan) → gen_image(step)]×k → gen_text(answer)
            for i in range(1, num_images + 1):
                # Generate step image with planning prompt
                if task_type == "maze":
                    plan_suffix = f"Step {i}: Generate an image showing the next move (one step up/down/left/right)."
                else:  # sliding
                    plan_suffix = f"Step {i}: Generate an image showing which tile to move and in which direction."

                gen_prompt = prompt + "\n\n" + plan_suffix

                _, img_paths = self._stage1_generate_image(
                    generation_prompt=gen_prompt,
                    doc_id=f"{doc_id}_step_{i:04d}",
                    task=task,
                    original_images=original_images,
                )

                if img_paths:
                    generated_images.extend(img_paths)
                    eval_logger.info(f"Saved step {i} image: {img_paths[0]}")

            # Final answer using all generated step images
            final_suffix = (
                "After the images, emit EXACTLY ONE LINE containing ONLY the final move list "
                "as <ANSWER_JSON>[...]</ANSWER_JSON>. No other text."
            )
            final_question = prompt + "\n\n" + final_suffix

            # Use custom stage 2 logic with all step images
            if generated_images:
                # Load all generated images
                step_images = [
                    Image.open(img_path).convert("RGB") for img_path in generated_images
                ]

                # Build images list
                images = []
                if original_image:
                    images.append(original_image)
                images.extend(step_images)

                # Normalize image sizes to ensure consistent dimensions
                images = self._normalize_image_sizes(images)

                # Build conversation format
                conversation = [
                    {
                        "role": "system",
                        "content": [
                            {"type": "text", "text": "You are a helpful assistant."}
                        ],
                    },
                    {
                        "role": "user",
                        "content": (
                            [{"type": "image"}] * len(images)
                            + [{"type": "text", "text": final_question}]
                        ),
                    },
                ]

                # Process inputs
                inputs = self._processor(
                    text=conversation, images=images, return_tensors="pt"
                )

                # Move inputs to appropriate device
                if self.infer_auto_device_map or self.device_map == "auto":
                    inputs = inputs.to("cuda")
                else:
                    inputs = inputs.to(self._device)

                # Generate answer
                with torch.no_grad():
                    output_ids = self._model.generate(
                        **inputs,
                        max_new_tokens=self.stage2_max_new_tokens,
                        do_sample=False,
                    )

                # Decode output
                input_len = inputs["input_ids"].shape[1]
                generated_ids = output_ids[0][input_len:]
                final_text = self.tokenizer.decode(
                    generated_ids, skip_special_tokens=True
                )

                # Clean up
                del inputs, output_ids, generated_ids
                torch.cuda.empty_cache()
            else:
                final_text = ""

        return final_text, generated_images

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(
            total=len(requests),
            disable=(self.rank != 0),
            desc="ILLUME+ Visual CoT",
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
            doc_id = doc_id[0]
            contexts = contexts[0]
            gen_kwargs = all_gen_kwargs[0]

            bagel_interleaved = gen_kwargs.get("bagel_interleaved", None)

            if bagel_interleaved is not None:
                doc = self.task_dict[task][split][doc_id]
                input_images = []
                if doc_to_visual[0]:
                    visuals = doc_to_visual[0](doc)
                    if visuals:
                        input_images = (
                            visuals if isinstance(visuals, list) else [visuals]
                        )

                final_answer, generated_images = self.generate_uni_mmmu_interleaved(
                    input_images, contexts, str(doc_id), task, bagel_interleaved, doc
                )

                self._save_intermediate_artifacts(
                    doc_id=str(doc_id),
                    task=task,
                    generation_prompt=f"Interleaved generation",
                    stage1_text="",
                    generated_images=generated_images,
                    question=contexts,
                    stage2_answer=final_answer,
                )

                res.append(final_answer)
                self.cache_hook.add_partial(
                    "generate_until", (contexts, gen_kwargs), final_answer
                )
                pbar.update(1)
                continue

            original_images = []
            if doc_to_visual[0]:
                try:
                    visuals = doc_to_visual[0](self.task_dict[task][split][doc_id])
                    if visuals:
                        original_images = (
                            visuals if isinstance(visuals, list) else [visuals]
                        )
                except Exception as e:
                    eval_logger.warning(f"Failed to extract images: {e}")

            gen_prompt_match = re.search(r"\[GEN_PROMPT\](.*?)\[/GEN_PROMPT\]", contexts, re.DOTALL)
            question_match = re.search(r"\[QUESTION\](.*?)\[/QUESTION\]", contexts, re.DOTALL)

            if gen_prompt_match and question_match:
                custom_gen_prompt = gen_prompt_match.group(1).strip()
                actual_question = question_match.group(1).strip()
                generation_prompt = custom_gen_prompt.replace("{question}", actual_question)
                contexts = contexts.replace(f"[GEN_PROMPT]{gen_prompt_match.group(1)}[/GEN_PROMPT]", "")
                contexts = contexts.replace(f"[QUESTION]{question_match.group(1)}[/QUESTION]", question_match.group(1))
            else:
                generation_prompt = self.generation_prompt_template.format(question=contexts)

            stage1_text, generated_images = self._stage1_generate_image(
                generation_prompt=generation_prompt,
                doc_id=doc_id,
                task=task,
                original_images=original_images,
            )

            if generated_images:
                final_answer = self._stage2_answer_with_images(
                    question=contexts,
                    generated_image_paths=generated_images,
                    original_images=original_images
                )
            else:
                final_answer = stage1_text if stage1_text else ""

            self._save_intermediate_artifacts(
                doc_id=doc_id,
                task=task,
                generation_prompt=generation_prompt,
                stage1_text=stage1_text,
                generated_images=generated_images,
                question=contexts,
                stage2_answer=final_answer,
            )

            res.append(final_answer)
            self.cache_hook.add_partial(
                "generate_until", (contexts, all_gen_kwargs[0]), final_answer
            )
            pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """Not supported for generation models."""
        raise NotImplementedError(
            "ILLUME+ Visual CoT is a generation model and does not support loglikelihood"
        )

    def generate_until_multi_round(self, requests) -> List[str]:
        """Multi-round dialogue not yet implemented."""
        raise NotImplementedError(
            "Multi-round dialogue not yet implemented for ILLUME+ Visual CoT"
        )
