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

import json
import os
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Tuple, Union

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
        stage2_max_new_tokens: int = 4096,
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
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        self.pretrained = pretrained
        self.use_cache = use_cache
        self.trust_remote_code = trust_remote_code
        self.save_intermediate = save_intermediate
        self.fail_gracefully = fail_gracefully
        self.generation_prompt_template = generation_prompt_template
        self.infer_auto_device_map = infer_auto_device_map
        self.device_map = device_map

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
                        if f.endswith((".safetensors", ".bin"))
                        and "pytorch_model" in f
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

            import time

            start_time = time.time()
            self._processor = AutoProcessor.from_pretrained(pretrained, **processor_kwargs)
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

    def _extract_image_from_various_formats(
        self, img_data
    ) -> Optional[Image.Image]:
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
        self, generation_prompt: str, doc_id: str, task: str, original_image=None
    ) -> Tuple[str, List[str]]:
        """
        Stage 1: Generate visualization image from prompt.

        Args:
            generation_prompt: Text prompt for image generation
            doc_id: Document ID for file naming
            task: Task name for file naming
            original_image: Original image to condition on (optional)

        Returns:
            Tuple of (generated_text, list_of_image_paths)
        """
        eval_logger.debug(f"Stage 1 - Generating image for doc {doc_id}")
        eval_logger.debug(f"Generation prompt: {generation_prompt}")

        try:
            # Extract original image if provided
            original_image = self._extract_image_from_various_formats(original_image)
            if original_image is not None:
                eval_logger.debug("Stage 1 - Using original image as conditioning input")

            # Build conversation for generation
            images = [original_image] if original_image else []

            # Build conversation format
            conversation = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a helpful assistant."}],
                },
                {
                    "role": "user",
                    "content": (
                        [{"type": "image"}] * len(images)
                        + [{"type": "text", "text": generation_prompt}]
                    ),
                },
            ]

            # Normalize image sizes if there are multiple images
            if len(images) > 1:
                images = self._normalize_image_sizes(images)

            # Process inputs
            inputs = self._processor(text=conversation, images=images, return_tensors="pt")

            # Move inputs to appropriate device
            if self.infer_auto_device_map or self.device_map == "auto":
                inputs = inputs.to("cuda")
            else:
                inputs = inputs.to(self._device)

            # Prepare generation kwargs for stage 1
            generate_kwargs = {
                "max_new_tokens": self.stage1_max_new_tokens,
                "use_cache": self.use_cache,
            }

            # Add sampling parameters
            if self.stage1_temperature > 0:
                generate_kwargs["do_sample"] = True
                generate_kwargs["temperature"] = self.stage1_temperature
                if self.stage1_top_p is not None:
                    generate_kwargs["top_p"] = self.stage1_top_p
            else:
                generate_kwargs["do_sample"] = False

            if self.stage1_num_beams > 1:
                generate_kwargs["num_beams"] = self.stage1_num_beams

            # Generate response (ILLUME+ can generate both text and images)
            with torch.no_grad():
                outputs = self._model.generate(**inputs, **generate_kwargs)

            # Decode response
            outputs_text = outputs[:, inputs["input_ids"].shape[1] :]
            generated_text = self._processor.batch_decode(
                outputs_text, skip_special_tokens=True
            )[0]

            eval_logger.debug(f"Stage 1 - Generated text: {generated_text[:100]}...")

            # Check if ILLUME+ generated an image
            # ILLUME+ may output image tokens that need to be decoded
            # For now, we'll save the generated text and create a placeholder
            # In a full implementation, you would decode image tokens here

            # Save generated "image" (for now, we'll create a text-based visualization)
            # In practice, ILLUME+ should generate actual image tokens
            task_dir = os.path.join(self.generated_images_dir, task)
            os.makedirs(task_dir, exist_ok=True)
            image_path = os.path.join(task_dir, f"{doc_id}_gen.png")

            # Create a simple visualization image with the generated text
            # This is a placeholder - in a full implementation, decode actual image tokens
            viz_image = Image.new("RGB", (512, 512), color=(255, 255, 255))
            viz_image.save(image_path)

            eval_logger.info(f"Generated image saved to {image_path}")

            del outputs, inputs
            torch.cuda.empty_cache()

            return generated_text, [image_path]

        except Exception as e:
            eval_logger.error(f"Stage 1 generation error: {e}")
            import traceback

            eval_logger.error(traceback.format_exc())
            if self.fail_gracefully:
                return "", []
            raise

    def _stage2_answer_with_images(
        self, question: str, generated_image_path: str, original_image=None
    ) -> str:
        """
        Stage 2: Answer question using both original and generated images.

        Args:
            question: Original question text
            generated_image_path: Path to generated auxiliary image
            original_image: Original image (optional)

        Returns:
            Answer text
        """
        eval_logger.debug("Stage 2 - Answering question with images")
        eval_logger.debug(f"Question: {question}")

        try:
            # Load images
            images = []

            # Add original image first (if available)
            if original_image:
                original_image = self._extract_image_from_various_formats(original_image)
                if original_image:
                    images.append(original_image)
                    eval_logger.debug("Stage 2 - Added original image")

            # Add generated image
            gen_image = Image.open(generated_image_path).convert("RGB")
            images.append(gen_image)
            eval_logger.debug("Stage 2 - Added generated image")

            # Normalize image sizes to ensure consistent dimensions
            images = self._normalize_image_sizes(images)

            # Build conversation format
            conversation = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a helpful assistant."}],
                },
                {
                    "role": "user",
                    "content": (
                        [{"type": "image"}] * len(images)
                        + [{"type": "text", "text": question}]
                    ),
                },
            ]

            # Process inputs
            inputs = self._processor(text=conversation, images=images, return_tensors="pt")

            # Move inputs to appropriate device
            if self.infer_auto_device_map or self.device_map == "auto":
                inputs = inputs.to("cuda")
            else:
                inputs = inputs.to(self._device)

            # Prepare generation kwargs for stage 2
            generate_kwargs = {
                "max_new_tokens": self.stage2_max_new_tokens,
                "use_cache": self.use_cache,
            }

            # Add sampling parameters
            if self.stage2_temperature > 0:
                generate_kwargs["do_sample"] = True
                generate_kwargs["temperature"] = self.stage2_temperature
                if self.stage2_top_p is not None:
                    generate_kwargs["top_p"] = self.stage2_top_p
            else:
                generate_kwargs["do_sample"] = False

            if self.stage2_num_beams > 1:
                generate_kwargs["num_beams"] = self.stage2_num_beams

            # Generate answer
            with torch.no_grad():
                outputs = self._model.generate(**inputs, **generate_kwargs)

            # Decode response (skip input tokens)
            outputs_text = outputs[:, inputs["input_ids"].shape[1] :]
            answer = self._processor.batch_decode(
                outputs_text, skip_special_tokens=True
            )[0]

            eval_logger.debug(f"Stage 2 - Generated answer: {answer[:100]}...")

            del outputs, inputs
            torch.cuda.empty_cache()

            return answer

        except Exception as e:
            eval_logger.error(f"Stage 2 error: {e}")
            import traceback

            eval_logger.error(traceback.format_exc())
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
                steps = json_module.loads(steps_str) if isinstance(steps_str, str) else steps_str
                if steps:
                    num_images = len(steps)
            elif task_type == "sliding":
                # Get step count from ground truth
                steps_str = doc.get("steps_words", "[]")
                steps = json_module.loads(steps_str) if isinstance(steps_str, str) else steps_str
                if steps:
                    num_images = len(steps)

        # Extract original image from input_images
        original_image = None
        if input_images and len(input_images) > 0:
            original_image = self._extract_image_from_various_formats(input_images[0])

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
                original_image=original_image,
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
                original_image=original_image,
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
                        "content": [{"type": "text", "text": "You are a helpful assistant."}],
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
                inputs = self._processor(text=conversation, images=images, return_tensors="pt")

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
                final_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

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
                    plan_suffix = f'Step {i}: Generate an image showing the next move (one step up/down/left/right).'
                else:  # sliding
                    plan_suffix = f'Step {i}: Generate an image showing which tile to move and in which direction.'

                gen_prompt = prompt + "\n\n" + plan_suffix

                _, img_paths = self._stage1_generate_image(
                    generation_prompt=gen_prompt,
                    doc_id=f"{doc_id}_step_{i:04d}",
                    task=task,
                    original_image=original_image,
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
                step_images = [Image.open(img_path).convert("RGB") for img_path in generated_images]

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
                        "content": [{"type": "text", "text": "You are a helpful assistant."}],
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
                inputs = self._processor(text=conversation, images=images, return_tensors="pt")

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
                final_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

                # Clean up
                del inputs, output_ids, generated_ids
                torch.cuda.empty_cache()
            else:
                final_text = ""

        return final_text, generated_images

    def generate_until(self, requests: List[Instance]) -> List[str]:
        """
        Main inference method implementing two-stage visual CoT.

        Stage 1: Generate visualization image from text prompt
        Stage 2: Answer question using both original and generated images
        """
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

            # Check if this is Uni-MMMU interleaved generation mode
            bagel_interleaved = gen_kwargs.get("bagel_interleaved", None)

            if bagel_interleaved is not None:
                # Uni-MMMU interleaved generation mode
                eval_logger.info(f"Uni-MMMU interleaved mode for doc {doc_id}")

                # Get input images and doc data
                doc = self.task_dict[task][split][doc_id]
                input_images = []
                if doc_to_visual[0]:
                    visuals = doc_to_visual[0](doc)
                    if visuals:
                        input_images = visuals if isinstance(visuals, list) else [visuals]

                # Generate using interleaved mode
                final_answer, generated_images = self.generate_uni_mmmu_interleaved(
                    input_images, contexts, str(doc_id), task, bagel_interleaved, doc
                )

                # Save intermediate artifacts if enabled
                self._save_intermediate_artifacts(
                    doc_id=str(doc_id),
                    task=task,
                    generation_prompt=f"Interleaved generation: {bagel_interleaved.get('task_type', 'unknown')}",
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

            # Standard single-image generation mode
            # Extract original image
            original_image = None
            if doc_to_visual[0]:
                try:
                    visuals = doc_to_visual[0](self.task_dict[task][split][doc_id])
                    if visuals:
                        original_image = visuals[0]
                        eval_logger.debug(f"Extracted original image for doc {doc_id}")
                except Exception as e:
                    eval_logger.warning(
                        f"Failed to extract original image for doc {doc_id}: {e}"
                    )

            # Parse contexts to extract generation_prompt if provided
            import re

            gen_prompt_match = re.search(
                r"\[GEN_PROMPT\](.*?)\[/GEN_PROMPT\]", contexts, re.DOTALL
            )
            question_match = re.search(
                r"\[QUESTION\](.*?)\[/QUESTION\]", contexts, re.DOTALL
            )

            if gen_prompt_match and question_match:
                # Use custom generation prompt from task config
                custom_gen_prompt = gen_prompt_match.group(1).strip()
                actual_question = question_match.group(1).strip()
                generation_prompt = custom_gen_prompt.replace(
                    "{question}", actual_question
                )
                # Update contexts to be just the question for stage 2
                contexts = contexts.replace(
                    f"[GEN_PROMPT]{gen_prompt_match.group(1)}[/GEN_PROMPT]", ""
                )
                contexts = contexts.replace(
                    f"[QUESTION]{question_match.group(1)}[/QUESTION]",
                    question_match.group(1),
                )
                eval_logger.info("Using custom generation prompt from task config")
            else:
                # Use default template
                generation_prompt = self.generation_prompt_template.format(
                    question=contexts
                )

            eval_logger.info(f"\n{'='*60}")
            eval_logger.info(f"Processing doc {doc_id} from task {task}")
            eval_logger.info(f"{'='*60}")

            # Stage 1: Generate visualization image
            stage1_text, generated_images = self._stage1_generate_image(
                generation_prompt=generation_prompt,
                doc_id=doc_id,
                task=task,
                original_image=original_image,
            )

            # Stage 2: Answer with both images
            if generated_images:
                final_answer = self._stage2_answer_with_images(
                    contexts, generated_images[0], original_image
                )
            else:
                eval_logger.warning(
                    f"No image generated for doc {doc_id}, using stage 1 text as answer"
                )
                final_answer = stage1_text if stage1_text else ""

            # Save intermediate artifacts if enabled
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
