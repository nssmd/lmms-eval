"""
UAE Visual Chain-of-Thought Model

Two-stage inference with memory-efficient loading:
1. Stage 1: Load generation components → Generate auxiliary image → Unload SD3
2. Stage 2: Use Qwen encoder (already loaded) → Answer question

Usage:
    python -m lmms_eval \
        --model uae_visual_cot \
        --model_args pretrained=zhiyuanyan1/UAE \
        --tasks illusionbench_arshia_logo_shape_visual_cot \
        --batch_size 1 \
        --device cuda:0 \
        --output_path ./logs/
"""

import gc
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

# Add UAE repository to Python path
wd = Path(__file__).parent.parent.parent.parent.resolve()
possible_paths = [
    os.path.join(str(wd), "UAE"),
    os.path.join(str(wd.parent), "UAE"),
    os.path.expanduser("~/data/zwb/UAE"),
    "/home/aiscuser/data/zwb/UAE",
]

uae_repo_path = None
for path in possible_paths:
    if os.path.exists(path):
        uae_repo_path = path
        break

if uae_repo_path:
    sys.path.insert(0, uae_repo_path)
    eval_logger.info(f"Added UAE repo path to sys.path: {uae_repo_path}")
else:
    eval_logger.warning(
        f"UAE repository not found. Tried: {possible_paths}. "
        f"Please clone it: git clone https://github.com/PKU-YuanGroup/UAE.git"
    )


@register_model("uae_visual_cot")
class UAEVisualCoT(lmms):
    """
    UAE Visual Chain-of-Thought Model with memory-efficient staged loading.

    Performs two-stage visual reasoning:
    1. Generate auxiliary visualization image from original image + text prompt
    2. Answer question using original image + auxiliary image + question
    """

    def __init__(
        self,
        pretrained: str = "zhiyuanyan1/UAE",
        # Stage 1: Image generation parameters
        stage1_num_inference_steps: int = 40,
        stage1_guidance_scale: float = 5.0,
        stage1_height: int = 1024,
        stage1_width: int = 1024,
        # Stage 2: Visual understanding parameters
        stage2_max_new_tokens: int = 512,
        stage2_temperature: float = 0.7,
        stage2_do_sample: bool = True,
        # Generation prompt template
        generation_prompt_template: str = "Generate a detailed visual diagram or illustration to help answer this question: {question}",
        # Output and debugging
        output_dir: Optional[str] = None,
        save_intermediate: bool = False,
        intermediate_dir: Optional[str] = None,
        # Error handling
        fail_gracefully: bool = True,
        # Model loading
        device: str = "cuda",
        dtype: str = "bfloat16",
        seed: int = 42,
        **kwargs,
    ) -> None:
        super().__init__()

        self.pretrained = pretrained
        self.save_intermediate = save_intermediate
        self.fail_gracefully = fail_gracefully
        self.seed = seed
        self.generation_prompt_template = generation_prompt_template

        # Stage 1 parameters
        self.stage1_num_inference_steps = stage1_num_inference_steps
        self.stage1_guidance_scale = stage1_guidance_scale
        self.stage1_height = stage1_height
        self.stage1_width = stage1_width

        # Stage 2 parameters
        self.stage2_max_new_tokens = stage2_max_new_tokens
        self.stage2_temperature = stage2_temperature
        self.stage2_do_sample = stage2_do_sample

        # Device and dtype
        self._device = torch.device(device if torch.cuda.is_available() else "cpu")
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        self._dtype = dtype_map.get(dtype, torch.bfloat16)

        # Setup output directories
        if output_dir is None:
            self.output_dir = "./logs/uae_visual_cot"
        else:
            self.output_dir = output_dir

        if intermediate_dir is None:
            self.intermediate_dir = self.output_dir
        else:
            self.intermediate_dir = intermediate_dir

        if save_intermediate:
            os.makedirs(self.intermediate_dir, exist_ok=True)
            eval_logger.info(
                f"Intermediate artifacts will be saved under: {self.intermediate_dir}"
            )

        # Setup distributed
        self._rank = 0
        self._world_size = 1

        # Set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Model components (loaded on demand)
        self._uae_model = None  # Qwen + Projector
        self._processor = None
        self._pipe = None  # SD3 Pipeline
        self._pooled_prompt_embeds = None

        # Paths (resolved once)
        self._model_paths = None

        # Initialize paths
        self._resolve_model_paths()

        eval_logger.info("UAEVisualCoT initialized (models will be loaded on demand)")

    def _resolve_model_paths(self):
        """Resolve model paths from pretrained argument"""
        if os.path.isdir(self.pretrained):
            self._model_paths = {
                "llm": os.path.join(self.pretrained, "llm_model"),
                "llm_lora": os.path.join(self.pretrained, "llm_lora"),
                "llm_processor": os.path.join(self.pretrained, "llm_model"),
                "SD3": os.path.join(self.pretrained, "SD3"),
                "dit": os.path.join(self.pretrained, "dit"),
                "dit_lora": os.path.join(self.pretrained, "dit_lora"),
            }
        else:
            from huggingface_hub import snapshot_download

            eval_logger.info(f"Downloading model from HuggingFace: {self.pretrained}")
            cache_dir = snapshot_download(repo_id=self.pretrained)
            self._model_paths = {
                "llm": os.path.join(cache_dir, "llm_model"),
                "llm_lora": os.path.join(cache_dir, "llm_lora"),
                "llm_processor": "Qwen/Qwen2.5-VL-3B-Instruct",  # Fallback
                "SD3": os.path.join(cache_dir, "SD3"),
                "dit": os.path.join(cache_dir, "dit"),
                "dit_lora": os.path.join(cache_dir, "dit_lora"),
            }
            # Check if processor files exist
            merges_file = os.path.join(cache_dir, "llm_model", "merges.txt")
            if os.path.exists(merges_file):
                self._model_paths["llm_processor"] = os.path.join(
                    cache_dir, "llm_model"
                )

    def _load_qwen_encoder(self):
        """Load Qwen encoder with projector (shared between Stage 1 and Stage 2)"""
        if self._uae_model is not None:
            return  # Already loaded

        eval_logger.info("Loading Qwen encoder with projector...")

        # Debug: check sys.path
        eval_logger.info(f"UAE repo in sys.path: {any('UAE' in p for p in sys.path)}")

        try:
            from uae.models.modeling_longcontext import Qwen2_5_VLWithLongContext
            from transformers import AutoProcessor
            from peft import LoraConfig, get_peft_model
            from safetensors.torch import load_file
        except ImportError as e:
            eval_logger.error(f"sys.path: {sys.path}")
            raise ImportError(
                f"UAE requires transformers, peft, and UAE repo. Error: {e}"
            )

        # Load model
        eval_logger.info(f"Loading from path: {self._model_paths['llm']}")
        eval_logger.info(f"Path exists: {os.path.exists(self._model_paths['llm'])}")

        try:
            self._uae_model = Qwen2_5_VLWithLongContext.from_pretrained(
                self._model_paths["llm"],
                torch_dtype=self._dtype,
                trust_remote_code=True,  # May be needed for custom config
            ).to(self._device)
        except Exception as e:
            eval_logger.error(f"Failed to load model: {e}")
            eval_logger.error(f"Model path: {self._model_paths['llm']}")
            import traceback

            traceback.print_exc()
            raise

        # Apply LoRA
        lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            init_lora_weights="gaussian",
            target_modules="all-linear",
        )
        self._uae_model.model = get_peft_model(self._uae_model.model, lora_config)

        # Load LoRA weights
        state_dict = {}
        lora_path = self._model_paths["llm_lora"]
        for i in range(1, 4):
            safe_tensor_path = os.path.join(
                lora_path, f"model-0000{i}-of-00003.safetensors"
            )
            if os.path.exists(safe_tensor_path):
                weights = load_file(safe_tensor_path)
                state_dict.update(weights)

        if state_dict:
            self._uae_model.load_state_dict(state_dict, strict=False)
            eval_logger.info("Loaded LLM LoRA weights")

        # Load processor
        self._processor = AutoProcessor.from_pretrained(
            self._model_paths["llm_processor"],
            trust_remote_code=True,
        )

        self._uae_model.eval()
        eval_logger.info("Qwen encoder loaded successfully")

    def _load_sd3_pipeline(self):
        """Load SD3 pipeline for image generation (Stage 1 only)"""
        if self._pipe is not None:
            return  # Already loaded

        eval_logger.info("Loading SD3 pipeline for Stage 1...")

        from diffusers import (
            StableDiffusion3Pipeline,
            SD3Transformer2DModel,
            AutoencoderKL,
        )
        from transformers import (
            CLIPTextModelWithProjection,
            T5EncoderModel,
            CLIPTokenizer,
            T5TokenizerFast,
        )
        from peft import PeftModel

        sd3_path = self._model_paths["SD3"]
        dit_path = self._model_paths["dit"]
        dit_lora_path = self._model_paths["dit_lora"]

        # Step 1: Load text encoders to compute pooled_prompt_embeds
        eval_logger.info("Loading text encoders for pooled embeddings...")

        text_encoder_1 = CLIPTextModelWithProjection.from_pretrained(
            sd3_path, subfolder="text_encoder", torch_dtype=self._dtype
        ).to(self._device)
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            sd3_path, subfolder="text_encoder_2", torch_dtype=self._dtype
        ).to(self._device)
        text_encoder_3 = T5EncoderModel.from_pretrained(
            sd3_path, subfolder="text_encoder_3", torch_dtype=self._dtype
        ).to(self._device)

        tokenizer_1 = CLIPTokenizer.from_pretrained(sd3_path, subfolder="tokenizer")
        tokenizer_2 = CLIPTokenizer.from_pretrained(sd3_path, subfolder="tokenizer_2")
        tokenizer_3 = T5TokenizerFast.from_pretrained(sd3_path, subfolder="tokenizer_3")

        # Compute pooled_prompt_embeds for empty prompt
        eval_logger.info("Computing pooled prompt embeddings...")
        try:
            from uae.utils.denoiser_prompt_embeds import encode_prompt

            _, _, self._pooled_prompt_embeds = encode_prompt(
                text_encoders=[text_encoder_1, text_encoder_2, text_encoder_3],
                tokenizers=[tokenizer_1, tokenizer_2, tokenizer_3],
                prompt="",
                max_sequence_length=512,
                device=self._device,
                num_images_per_prompt=1,
            )
        except ImportError:
            # Fallback: compute pooled embeddings manually
            eval_logger.warning(
                "encode_prompt not found, computing pooled embeddings manually"
            )
            with torch.no_grad():
                text_input_1 = tokenizer_1(
                    "", padding="max_length", max_length=77, return_tensors="pt"
                ).to(self._device)
                text_input_2 = tokenizer_2(
                    "", padding="max_length", max_length=77, return_tensors="pt"
                ).to(self._device)
                pooled_1 = text_encoder_1(text_input_1.input_ids).text_embeds
                pooled_2 = text_encoder_2(text_input_2.input_ids).text_embeds
                self._pooled_prompt_embeds = torch.cat([pooled_1, pooled_2], dim=-1)

        # Step 2: Unload text encoders to free memory
        eval_logger.info("Unloading text encoders to free memory...")
        del text_encoder_1, text_encoder_2, text_encoder_3
        del tokenizer_1, tokenizer_2, tokenizer_3
        gc.collect()
        torch.cuda.empty_cache()

        # Step 3: Load transformer and VAE
        eval_logger.info("Loading SD3 transformer and VAE...")
        transformer = SD3Transformer2DModel.from_pretrained(
            dit_path, torch_dtype=self._dtype
        ).to(self._device)

        vae = AutoencoderKL.from_pretrained(
            sd3_path, subfolder="vae", torch_dtype=self._dtype
        ).to(self._device)

        # Step 4: Create pipeline
        self._pipe = StableDiffusion3Pipeline.from_pretrained(
            sd3_path,
            vae=vae,
            transformer=transformer,
            text_encoder=None,
            text_encoder_2=None,
            text_encoder_3=None,
            torch_dtype=self._dtype,
        ).to(self._device)

        # Apply LoRA to transformer
        if os.path.exists(dit_lora_path):
            self._pipe.transformer = PeftModel.from_pretrained(
                self._pipe.transformer, dit_lora_path
            )
            eval_logger.info("Loaded DiT LoRA weights")

        eval_logger.info("SD3 pipeline loaded successfully")

    def _unload_sd3_pipeline(self):
        """Unload SD3 pipeline to free memory for Stage 2"""
        if self._pipe is None:
            return

        eval_logger.info("Unloading SD3 pipeline to free memory...")
        del self._pipe
        self._pipe = None
        # Keep pooled_prompt_embeds as it's small
        gc.collect()
        torch.cuda.empty_cache()
        eval_logger.info("SD3 pipeline unloaded")

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    @property
    def model(self):
        return self._uae_model

    @property
    def tokenizer(self):
        return self._processor.tokenizer if self._processor else None

    def _get_prompt_embeds(
        self, prompt: str, image: Optional[Image.Image] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get prompt embeddings using Qwen encoder + projector.

        Args:
            prompt: Text prompt for generation
            image: Optional input image to condition on

        Returns:
            Tuple of (negative_prompt_embeds, positive_prompt_embeds)
        """
        # Prepare negative prompt (text only)
        negative_messages = [
            {
                "role": "generate",
                "content": [
                    {
                        "type": "text",
                        "text": "Generate a random, low quality, ugly, blur, bad and anime, cartoon image.",
                    }
                ],
            }
        ]

        # Prepare positive prompt (with optional image)
        positive_content = []
        if image is not None:
            positive_content.append({"type": "image", "image": image})
        positive_content.append({"type": "text", "text": prompt})

        positive_messages = [
            {
                "role": "generate",
                "content": positive_content,
            }
        ]

        # Process negative prompt
        negative_text = self._processor.apply_chat_template(
            negative_messages, tokenize=False, add_generation_prompt=True
        )
        negative_inputs = self._processor(
            text=[negative_text],
            images=None,
            videos=None,
            return_tensors="pt",
            padding=True,
        ).to(self._device)

        # Process positive prompt
        positive_text = self._processor.apply_chat_template(
            positive_messages, tokenize=False, add_generation_prompt=True
        )

        try:
            from qwen_vl_utils import process_vision_info

            image_inputs, video_inputs = process_vision_info(positive_messages)
        except ImportError:
            image_inputs = [image] if image else None
            video_inputs = None

        positive_inputs = self._processor(
            text=[positive_text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
        ).to(self._device)

        # Get projected embeddings
        with torch.no_grad():
            negative_embeds = self._uae_model.get_projected_embeddings(
                input_ids=negative_inputs.input_ids,
                attention_mask=negative_inputs.attention_mask,
            )

            positive_embeds = self._uae_model.get_projected_embeddings(
                input_ids=positive_inputs.input_ids,
                attention_mask=positive_inputs.attention_mask,
                pixel_values=positive_inputs.get("pixel_values"),
                image_grid_thw=positive_inputs.get("image_grid_thw"),
            )

        # Pad embeddings to same length (SD3 requires this)
        max_len = max(negative_embeds.shape[1], positive_embeds.shape[1])
        if negative_embeds.shape[1] < max_len:
            padding = torch.zeros(
                negative_embeds.shape[0],
                max_len - negative_embeds.shape[1],
                negative_embeds.shape[2],
                dtype=negative_embeds.dtype,
                device=negative_embeds.device,
            )
            negative_embeds = torch.cat([negative_embeds, padding], dim=1)
        if positive_embeds.shape[1] < max_len:
            padding = torch.zeros(
                positive_embeds.shape[0],
                max_len - positive_embeds.shape[1],
                positive_embeds.shape[2],
                dtype=positive_embeds.dtype,
                device=positive_embeds.device,
            )
            positive_embeds = torch.cat([positive_embeds, padding], dim=1)

        return negative_embeds, positive_embeds

    def _stage1_generate_image(
        self,
        generation_prompt: str,
        doc_id: str,
        task: str,
        original_image: Optional[Image.Image] = None,
    ) -> Tuple[str, List[str]]:
        """
        Stage 1: Generate auxiliary visualization image.

        Memory-efficient approach:
        1. Load Qwen → generate prompt_embeds → unload Qwen
        2. Load SD3 → use prompt_embeds → generate image

        Args:
            generation_prompt: Text prompt for generation
            doc_id: Document ID
            task: Task name
            original_image: Optional original image to condition on

        Returns:
            Tuple of (prompt_text, list_of_generated_image_paths)
        """
        eval_logger.info(f"Stage 1 - Generating image for doc {doc_id}")
        if original_image:
            eval_logger.info("Using original image as conditioning input")

        try:
            # Step 1: Load Qwen encoder and generate prompt embeddings
            eval_logger.info("Step 1/3: Loading Qwen encoder for prompt embeddings...")
            self._load_qwen_encoder()
            negative_embeds, positive_embeds = self._get_prompt_embeds(
                generation_prompt, image=original_image
            )
            eval_logger.info(f"Generated prompt_embeds: {positive_embeds.shape}")

            # Step 2: Unload Qwen to free memory for SD3
            eval_logger.info("Step 2/3: Unloading Qwen encoder to free memory...")
            del self._uae_model
            del self._processor
            self._uae_model = None
            self._processor = None
            gc.collect()
            torch.cuda.empty_cache()

            # Step 3: Load SD3 pipeline and generate image
            eval_logger.info("Step 3/3: Loading SD3 pipeline for image generation...")
            self._load_sd3_pipeline()

            # Generate image
            generator = torch.Generator(device=self._device).manual_seed(self.seed)

            with torch.no_grad():
                result = self._pipe(
                    prompt_embeds=positive_embeds,
                    negative_prompt_embeds=negative_embeds,
                    pooled_prompt_embeds=self._pooled_prompt_embeds,
                    negative_pooled_prompt_embeds=self._pooled_prompt_embeds,
                    height=self.stage1_height,
                    width=self.stage1_width,
                    num_inference_steps=self.stage1_num_inference_steps,
                    guidance_scale=self.stage1_guidance_scale,
                    generator=generator,
                )
                generated_image = result.images[0]

            # Save image
            task_dir = os.path.join(self.intermediate_dir, task)
            os.makedirs(task_dir, exist_ok=True)
            image_path = os.path.join(task_dir, f"{doc_id}_stage1.png")
            generated_image.save(image_path)
            eval_logger.info(f"Stage 1 - Saved image: {image_path}")

            return generation_prompt, [image_path]

        except Exception as e:
            eval_logger.error(f"Stage 1 failed for doc {doc_id}: {e}")
            if self.fail_gracefully:
                return generation_prompt, []
            raise

    def _stage2_answer_with_images(
        self,
        question: str,
        auxiliary_image_path: str,
        doc_id: str,
        original_image: Optional[Image.Image] = None,
    ) -> str:
        """
        Stage 2: Answer question using original + auxiliary images.

        Re-load Qwen encoder if needed (SD3 was unloaded after Stage 1).

        Args:
            question: Question text
            auxiliary_image_path: Path to generated auxiliary image
            doc_id: Document ID
            original_image: Original image

        Returns:
            Answer text
        """
        eval_logger.info(f"Stage 2 - Answering question for doc {doc_id}")

        try:
            # Unload SD3 pipeline if still loaded
            self._unload_sd3_pipeline()

            # Re-load Qwen encoder for Stage 2
            eval_logger.info("Reloading Qwen encoder for Stage 2...")
            self._load_qwen_encoder()

            # Load auxiliary image
            auxiliary_image = Image.open(auxiliary_image_path).convert("RGB")

            # Prepare messages with images
            if original_image is not None:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "You are given two images: the original image and an auxiliary visualization. ",
                            },
                            {"type": "image", "image": original_image},
                            {
                                "type": "text",
                                "text": "Here is the auxiliary visualization: ",
                            },
                            {"type": "image", "image": auxiliary_image},
                            {"type": "text", "text": question},
                        ],
                    }
                ]
            else:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": auxiliary_image},
                            {"type": "text", "text": question},
                        ],
                    }
                ]

            # Process with Qwen
            text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            try:
                from qwen_vl_utils import process_vision_info

                image_inputs, video_inputs = process_vision_info(messages)
            except ImportError:
                image_inputs = (
                    [original_image, auxiliary_image]
                    if original_image
                    else [auxiliary_image]
                )
                video_inputs = None

            model_inputs = self._processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                return_tensors="pt",
                padding=True,
            ).to(self._device)

            # Generate answer
            with torch.no_grad():
                generated_ids = self._uae_model.generate(
                    **model_inputs,
                    max_new_tokens=self.stage2_max_new_tokens,
                    do_sample=self.stage2_do_sample,
                    temperature=self.stage2_temperature,
                )

            # Decode
            generated_ids = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            answer = self._processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]

            eval_logger.info(f"Stage 2 - Generated answer: {answer[:100]}...")
            return answer

        except Exception as e:
            eval_logger.error(f"Stage 2 failed for doc {doc_id}: {e}")
            if self.fail_gracefully:
                return ""
            raise

    def _extract_image(self, visual) -> Optional[Image.Image]:
        """Extract PIL Image from various formats"""
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

    def flatten(self, input_list: List) -> List:
        """Flatten a nested list"""
        output = []
        for item in input_list:
            if isinstance(item, list):
                output.extend(self.flatten(item))
            else:
                output.append(item)
        return output

    def _save_intermediate_artifacts(
        self,
        doc_id: str,
        task: str,
        generation_prompt: str,
        generated_images: List[str],
        question: str,
        stage2_answer: str,
    ) -> None:
        """Save intermediate artifacts for debugging"""
        if not self.save_intermediate:
            return

        artifact_dir = os.path.join(self.intermediate_dir, task)
        os.makedirs(artifact_dir, exist_ok=True)

        # Save metadata
        metadata = {
            "doc_id": doc_id,
            "task": task,
            "generation_prompt": generation_prompt,
            "generated_images": generated_images,
            "question": question,
            "stage2_answer": stage2_answer,
        }

        metadata_path = os.path.join(artifact_dir, f"{doc_id}_metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        eval_logger.debug(f"Saved intermediate artifacts to: {metadata_path}")

    def generate_until(self, requests: List[Instance]) -> List[str]:
        """Main inference method implementing two-stage visual CoT"""
        res = []
        pbar = tqdm(
            total=len(requests),
            disable=(self.rank != 0),
            desc="UAEVisualCoT Generating",
        )

        for request in requests:
            contexts, gen_kwargs, doc_to_visual, doc_id, task, split = request.args

            # Extract original image
            original_image = None
            if doc_to_visual is not None:
                try:
                    doc = self.task_dict[task][split][doc_id]
                    original_visuals = doc_to_visual(doc)
                    if original_visuals:
                        flattened = self.flatten([original_visuals])
                        if flattened:
                            original_image = self._extract_image(flattened[0])
                except Exception as e:
                    eval_logger.warning(f"Failed to extract original image: {e}")

            # Parse generation prompt
            import re

            gen_prompt_match = re.search(
                r"\[GEN_PROMPT\](.*?)\[/GEN_PROMPT\]", contexts, re.DOTALL
            )
            question_match = re.search(
                r"\[QUESTION\](.*?)\[/QUESTION\]", contexts, re.DOTALL
            )

            if gen_prompt_match and question_match:
                custom_gen_prompt = gen_prompt_match.group(1).strip()
                actual_question = question_match.group(1).strip()
                generation_prompt = custom_gen_prompt.replace(
                    "{question}", actual_question
                )
                contexts = actual_question
            else:
                generation_prompt = self.generation_prompt_template.format(
                    question=contexts
                )

            eval_logger.info(f"\n{'=' * 60}")
            eval_logger.info(f"Processing doc {doc_id} from task {task}")
            eval_logger.info(f"{'=' * 60}")

            # Stage 1: Generate auxiliary image (with original image conditioning)
            _, generated_images = self._stage1_generate_image(
                generation_prompt=generation_prompt,
                doc_id=str(doc_id),
                task=task,
                original_image=original_image,
            )

            if not generated_images:
                eval_logger.warning(f"No image generated for doc {doc_id}")
                res.append("")
                pbar.update(1)
                continue

            # Stage 2: Answer question
            final_answer = self._stage2_answer_with_images(
                question=contexts,
                auxiliary_image_path=generated_images[0],
                doc_id=str(doc_id),
                original_image=original_image,
            )

            # Save metadata
            if self.save_intermediate:
                self._save_intermediate_artifacts(
                    doc_id=str(doc_id),
                    task=task,
                    generation_prompt=generation_prompt,
                    generated_images=generated_images,
                    question=contexts,
                    stage2_answer=final_answer,
                )

            res.append(final_answer)
            pbar.update(1)

        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("UAEVisualCoT does not support loglikelihood")

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("Multi-round not implemented for UAEVisualCoT")
