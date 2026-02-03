"""
OmniGen2 Visual Chain-of-Thought Model

Two-stage inference:
1. Stage 1: Generate auxiliary visualization image from original image + text prompt
2. Stage 2: Answer question using original image + auxiliary image

Usage:
    python -m lmms_eval \
        --model omnigen2_visual_cot \
        --model_args pretrained=OmniGen2/OmniGen2 \
        --tasks mathvista_visual_cot \
        --batch_size 1 \
        --device cuda:0 \
        --output_path ./logs/
"""

import json
import os
import re
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

# Add OmniGen2 repository to Python path
wd = Path(__file__).parent.parent.parent.parent.resolve()
possible_paths = [
    os.path.join(str(wd), "OmniGen2"),
    os.path.join(str(wd.parent), "OmniGen2"),
    os.path.expanduser("~/data/zwb/OmniGen2"),
    "/home/aiscuser/data/zwb/OmniGen2",
]

omnigen2_path = None
for path in possible_paths:
    if os.path.exists(path):
        omnigen2_path = path
        break

if omnigen2_path:
    sys.path.insert(0, omnigen2_path)
    eval_logger.info(f"Added OmniGen2 path to sys.path: {omnigen2_path}")
else:
    eval_logger.warning(
        f"OmniGen2 repository not found. Tried: {possible_paths}. "
        f"Please clone it: git clone https://github.com/VectorSpaceLab/OmniGen2.git"
    )


@register_model("omnigen2_visual_cot")
class OmniGen2VisualCoT(lmms):
    """
    OmniGen2 Visual Chain-of-Thought Model

    Performs two-stage visual reasoning:
    1. Generate auxiliary visualization image from original image + prompt
    2. Answer question using both original and auxiliary images
    """

    def __init__(
        self,
        pretrained: str = "OmniGen2/OmniGen2",
        # Stage 1: Image generation parameters
        stage1_num_inference_steps: int = 50,
        stage1_text_guidance_scale: float = 5.0,
        stage1_image_guidance_scale: float = 2.0,
        stage1_cfg_range: Tuple[float, float] = (0.0, 1.0),
        stage1_height: int = 1024,
        stage1_width: int = 1024,
        stage1_negative_prompt: str = "blurry, low quality, text, watermark",
        # Stage 2: Visual understanding parameters
        stage2_max_new_tokens: int = 512,
        # Generation prompt template
        generation_prompt_template: str = "Based on the input image, generate a detailed visual diagram or illustration to help answer this question: {question}",
        # Output and debugging
        output_dir: Optional[str] = None,
        save_intermediate: bool = True,
        intermediate_dir: Optional[str] = None,
        # Error handling
        fail_gracefully: bool = True,
        # Memory optimization
        enable_model_cpu_offload: bool = False,
        enable_teacache: bool = False,
        teacache_thresh: float = 0.05,
        seed: int = 42,
        device: str = "cuda",
        dtype: str = "bfloat16",
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
        self.stage1_text_guidance_scale = stage1_text_guidance_scale
        self.stage1_image_guidance_scale = stage1_image_guidance_scale
        self.stage1_cfg_range = stage1_cfg_range
        self.stage1_height = stage1_height
        self.stage1_width = stage1_width
        self.stage1_negative_prompt = stage1_negative_prompt

        # Stage 2 parameters
        self.stage2_max_new_tokens = stage2_max_new_tokens

        # Memory optimization
        self.enable_model_cpu_offload = enable_model_cpu_offload
        self.enable_teacache = enable_teacache
        self.teacache_thresh = teacache_thresh

        # Determine device and dtype
        if torch.cuda.is_available():
            self._device = torch.device(device)
        else:
            self._device = torch.device("cpu")

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        self._dtype = dtype_map.get(dtype, torch.bfloat16)

        # Setup output directories
        if output_dir is None:
            self.output_dir = "./logs/omnigen2_visual_cot"
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

        # Load models
        eval_logger.info(f"Loading OmniGen2 model from {pretrained}")
        self._load_models()

        eval_logger.info("OmniGen2VisualCoT initialized successfully")

    def _load_models(self):
        """Load OmniGen2 pipelines for both generation and understanding"""
        try:
            from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
            from omnigen2.pipelines.omnigen2.pipeline_omnigen2_chat import (
                OmniGen2ChatPipeline,
            )
        except ImportError as e:
            raise ImportError(
                f"Failed to import OmniGen2. "
                f"Please ensure OmniGen2 repository is cloned. "
                f"Error: {e}"
            )

        # Load generation pipeline for Stage 1
        self._gen_pipeline = OmniGen2Pipeline.from_pretrained(
            self.pretrained,
            torch_dtype=self._dtype,
            trust_remote_code=True,
        )

        # Load chat pipeline for Stage 2 (understanding)
        self._chat_pipeline = OmniGen2ChatPipeline.from_pretrained(
            self.pretrained,
            torch_dtype=self._dtype,
            trust_remote_code=True,
        )

        # Apply memory optimizations
        if self.enable_model_cpu_offload:
            self._gen_pipeline.enable_model_cpu_offload()
            self._chat_pipeline.enable_model_cpu_offload()
            eval_logger.info("Enabled model CPU offload")
        else:
            self._gen_pipeline = self._gen_pipeline.to(self._device)
            self._chat_pipeline = self._chat_pipeline.to(self._device)

        # Ensure transformer has enable_teacache attribute (required by OmniGen2 pipeline)
        # The HuggingFace version may not have this attribute
        if hasattr(self._gen_pipeline, "transformer"):
            if not hasattr(self._gen_pipeline.transformer, "enable_teacache"):
                self._gen_pipeline.transformer.enable_teacache = False
        if hasattr(self._chat_pipeline, "transformer"):
            if not hasattr(self._chat_pipeline.transformer, "enable_teacache"):
                self._chat_pipeline.transformer.enable_teacache = False

        # Enable TeaCache for faster inference (only if supported)
        if self.enable_teacache:
            if hasattr(self._gen_pipeline, "transformer") and hasattr(
                self._gen_pipeline.transformer, "teacache_rel_l1_thresh"
            ):
                self._gen_pipeline.transformer.enable_teacache = True
                self._gen_pipeline.transformer.teacache_rel_l1_thresh = (
                    self.teacache_thresh
                )
                eval_logger.info(
                    f"Enabled TeaCache with threshold {self.teacache_thresh}"
                )
            else:
                eval_logger.warning(
                    "TeaCache not fully supported by this transformer version, skipping"
                )

        self._tokenizer = self._chat_pipeline.processor.tokenizer

        eval_logger.info("OmniGen2 pipelines loaded successfully")

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    @property
    def model(self):
        return self._gen_pipeline

    @property
    def tokenizer(self):
        return self._tokenizer

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

    def _stage1_generate_image(
        self,
        generation_prompt: str,
        doc_id: str,
        task: str,
        original_image: Optional[Image.Image] = None,
    ) -> Tuple[str, List[str]]:
        """
        Stage 1: Generate auxiliary visualization image

        Args:
            generation_prompt: Text prompt for image generation
            doc_id: Document ID for file naming
            task: Task name for file naming
            original_image: Original image to condition on

        Returns:
            Tuple of (prompt_used, list_of_image_paths)
        """
        eval_logger.debug(f"Stage 1 - Generating image for doc {doc_id}")
        eval_logger.debug(f"Generation prompt: {generation_prompt}")

        try:
            # Set seed for reproducibility
            generator = torch.Generator(device=self._device).manual_seed(self.seed)

            # Prepare input images
            input_images = [original_image] if original_image is not None else None

            # Generate auxiliary image
            with torch.no_grad():
                result = self._gen_pipeline(
                    prompt=generation_prompt,
                    input_images=input_images,
                    height=self.stage1_height,
                    width=self.stage1_width,
                    num_inference_steps=self.stage1_num_inference_steps,
                    text_guidance_scale=self.stage1_text_guidance_scale,
                    image_guidance_scale=self.stage1_image_guidance_scale,
                    cfg_range=self.stage1_cfg_range,
                    negative_prompt=self.stage1_negative_prompt,
                    generator=generator,
                    return_dict=True,
                )

            # Save generated images
            output_images = []
            if result.images:
                task_dir = os.path.join(self.intermediate_dir, task)
                os.makedirs(task_dir, exist_ok=True)

                for i, img in enumerate(result.images):
                    if i == 0:
                        safe_filename = f"{doc_id}_stage1.png"
                    else:
                        safe_filename = f"{doc_id}_stage1_{i}.png"
                    image_path = os.path.join(task_dir, safe_filename)
                    img.save(image_path)
                    output_images.append(image_path)
                    eval_logger.debug(f"Stage 1 - Saved image: {image_path}")

            eval_logger.debug(f"Stage 1 - Generated {len(output_images)} image(s)")
            return generation_prompt, output_images

        except Exception as e:
            eval_logger.error(f"Stage 1 failed for doc {doc_id}: {e}")
            if self.fail_gracefully:
                return generation_prompt, []
            else:
                raise

    def _stage2_answer_with_image(
        self,
        question: str,
        auxiliary_image_path: str,
        doc_id: str,
        original_image: Optional[Image.Image] = None,
    ) -> str:
        """
        Stage 2: Answer question using original + auxiliary images

        Args:
            question: Original question text
            auxiliary_image_path: Path to generated auxiliary image
            doc_id: Document ID for logging
            original_image: Original image

        Returns:
            Answer text
        """
        eval_logger.debug(f"Stage 2 - Answering question for doc {doc_id}")
        eval_logger.debug(f"Question: {question}")

        try:
            # Load auxiliary image
            auxiliary_image = Image.open(auxiliary_image_path).convert("RGB")

            # Prepare images for understanding
            # Use both original and auxiliary images if original is available
            if original_image is not None:
                images = [original_image, auxiliary_image]
                eval_logger.debug("Stage 2 - Using both original and auxiliary images")
            else:
                images = [auxiliary_image]
                eval_logger.debug("Stage 2 - Using auxiliary image only")

            # Use ChatPipeline's generate_text method
            formatted_prompt = self._chat_pipeline._apply_chat_template(
                question, images
            )
            output_texts = self._chat_pipeline.generate_text(formatted_prompt, images)

            if output_texts:
                answer_text = output_texts[0]
                # Clean up response
                answer_text = answer_text.replace("<|im_end|>", "").strip()
            else:
                answer_text = ""

            eval_logger.debug(f"Stage 2 - Generated answer: {answer_text[:100]}...")
            return answer_text

        except Exception as e:
            eval_logger.error(f"Stage 2 failed for doc {doc_id}: {e}")
            if self.fail_gracefully:
                return ""
            else:
                raise

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
        """
        Main inference method implementing two-stage visual CoT

        Stage 1: Generate auxiliary visualization image from original image + prompt
        Stage 2: Answer question using original + auxiliary images
        """
        res = []
        pbar = tqdm(
            total=len(requests),
            disable=(self.rank != 0),
            desc="OmniGen2VisualCoT Generating",
        )

        for request in requests:
            contexts, gen_kwargs, doc_to_visual, doc_id, task, split = request.args

            # Extract original image from document
            original_image = None
            if doc_to_visual is not None:
                try:
                    doc = self.task_dict[task][split][doc_id]
                    original_visuals = doc_to_visual(doc)
                    if original_visuals:
                        flattened = self.flatten([original_visuals])
                        if flattened:
                            original_image = self._extract_image(flattened[0])
                            eval_logger.debug(
                                f"Extracted original image for doc {doc_id}"
                            )
                except Exception as e:
                    eval_logger.warning(
                        f"Failed to extract original image for doc {doc_id}: {e}"
                    )

            # Parse contexts to extract generation_prompt if provided
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
                # Clean contexts for stage 2
                contexts = actual_question
                eval_logger.debug("Using custom generation prompt from task config")
            else:
                # Use default template
                generation_prompt = self.generation_prompt_template.format(
                    question=contexts
                )

            eval_logger.info(f"\n{'='*60}")
            eval_logger.info(f"Processing doc {doc_id} from task {task}")
            eval_logger.info(f"{'='*60}")

            # Stage 1: Generate auxiliary visualization image
            _, generated_images = self._stage1_generate_image(
                generation_prompt=generation_prompt,
                doc_id=str(doc_id),
                task=task,
                original_image=original_image,
            )

            # Check if image was generated
            if not generated_images:
                eval_logger.warning(
                    f"No image generated for doc {doc_id}, returning empty answer"
                )
                res.append("")
                pbar.update(1)
                continue

            # Stage 2: Answer question using original + auxiliary images
            final_answer = self._stage2_answer_with_image(
                question=contexts,
                auxiliary_image_path=generated_images[0],
                doc_id=str(doc_id),
                original_image=original_image,
            )

            # Save intermediate artifacts if enabled
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
        """Not supported for generation models"""
        raise NotImplementedError(
            "OmniGen2VisualCoT is a generation model and does not support loglikelihood"
        )

    def generate_until_multi_round(self, requests) -> List[str]:
        """Multi-round dialogue not yet implemented"""
        raise NotImplementedError(
            "Multi-round dialogue not yet implemented for OmniGen2VisualCoT"
        )
