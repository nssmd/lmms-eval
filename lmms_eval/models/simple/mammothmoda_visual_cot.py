"""
MammothModa Visual Chain-of-Thought Model

Two-stage inference:
1. Stage 1: Generate visualization image from text prompt
2. Stage 2: Answer question using the generated image

Usage:
    python -m lmms_eval \
        --model mammothmoda_visual_cot \
        --model_args pretrained=/path/to/MammothModa2-Preview \
        --tasks mathvista_visual_cot \
        --batch_size 1 \
        --device cuda:0 \
        --output_path ./logs/
"""

import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model


@register_model("mammothmoda_visual_cot")
class MammothModaVisualCoT(lmms):
    """
    MammothModa Visual Chain-of-Thought Model

    Performs two-stage visual reasoning:
    1. Generate visualization image from text prompt
    2. Answer question using the generated image
    """

    def __init__(
        self,
        pretrained: str,
        # Stage 1: Image generation parameters
        stage1_cfg_scale: float = 7.0,
        stage1_text_guidance_scale: float = 9.0,
        stage1_num_inference_steps: int = 50,
        stage1_height: int = 1024,
        stage1_width: int = 1024,
        stage1_ar_height: int = 32,
        stage1_ar_width: int = 32,
        # Stage 2: Visual understanding parameters
        stage2_max_new_tokens: int = 512,
        stage2_temperature: float = 0.0,
        stage2_do_sample: bool = False,
        # Generation prompt template
        generation_prompt_template: str = "Generate a detailed visual diagram or illustration to help answer this question: {question}",
        # Output and debugging
        output_dir: Optional[str] = None,
        save_intermediate: bool = False,
        intermediate_dir: Optional[str] = None,
        # Error handling
        fail_gracefully: bool = True,
        # Model loading
        attn_implementation: str = "flash_attention_2",
        torch_dtype: str = "bfloat16",
        seed: int = 0,
        **kwargs,
    ) -> None:
        super().__init__()

        self.pretrained = pretrained
        self.save_intermediate = save_intermediate
        self.fail_gracefully = fail_gracefully
        self.seed = seed
        self.generation_prompt_template = generation_prompt_template

        # Stage 1 parameters
        self.stage1_cfg_scale = stage1_cfg_scale
        self.stage1_text_guidance_scale = stage1_text_guidance_scale
        self.stage1_num_inference_steps = stage1_num_inference_steps
        self.stage1_height = stage1_height
        self.stage1_width = stage1_width
        self.stage1_ar_height = stage1_ar_height
        self.stage1_ar_width = stage1_ar_width

        # Stage 2 parameters
        self.stage2_max_new_tokens = stage2_max_new_tokens
        self.stage2_temperature = stage2_temperature
        self.stage2_do_sample = stage2_do_sample

        # Setup output directories
        if output_dir is None:
            self.output_dir = "./logs/mammothmoda_visual_cot"
        else:
            self.output_dir = output_dir

        if intermediate_dir is None:
            self.intermediate_dir = self.output_dir
        else:
            self.intermediate_dir = intermediate_dir

        if save_intermediate:
            os.makedirs(self.intermediate_dir, exist_ok=True)
            eval_logger.info(f"Intermediate artifacts will be saved under: {self.intermediate_dir}")

        # Store model loading parameters for dynamic loading
        self.attn_implementation = attn_implementation
        self.torch_dtype_str = torch_dtype

        # Model instances (will be loaded dynamically)
        self.mammothmoda_gen = None
        self.mammothmoda_und = None
        self.current_model_mode = None

        # Import MammothModa class for later use
        from lmms_eval.models.simple.mammothmoda import MammothModa
        self.MammothModa = MammothModa

        eval_logger.info("MammothModaVisualCoT initialized (models will be loaded dynamically to save memory)")

    def _load_generation_model(self):
        """Load generation model and unload understanding model"""
        import gc
        import torch

        if self.current_model_mode == "generation":
            return  # Already loaded

        eval_logger.info("Loading generation model...")

        # Unload understanding model if loaded
        if self.mammothmoda_und is not None:
            eval_logger.info("Unloading understanding model to free memory...")
            del self.mammothmoda_und
            self.mammothmoda_und = None
            gc.collect()
            torch.cuda.empty_cache()

        # Load generation model
        self.mammothmoda_gen = self.MammothModa(
            pretrained=self.pretrained,
            mode="generation",
            attn_implementation=self.attn_implementation,
            torch_dtype=self.torch_dtype_str,
            output_image_dir=self.intermediate_dir,
            cfg_scale=self.stage1_cfg_scale,
            text_guidance_scale=self.stage1_text_guidance_scale,
            num_inference_steps=self.stage1_num_inference_steps,
            height=self.stage1_height,
            width=self.stage1_width,
            ar_height=self.stage1_ar_height,
            ar_width=self.stage1_ar_width,
            seed=self.seed,
            continual_mode=False,
        )

        self.current_model_mode = "generation"
        eval_logger.info("Generation model loaded successfully")

    def _load_understanding_model(self):
        """Load understanding model and unload generation model"""
        import gc
        import torch

        if self.current_model_mode == "understanding":
            return  # Already loaded

        eval_logger.info("Loading understanding model...")

        # Unload generation model if loaded
        if self.mammothmoda_gen is not None:
            eval_logger.info("Unloading generation model to free memory...")
            del self.mammothmoda_gen
            self.mammothmoda_gen = None
            gc.collect()
            torch.cuda.empty_cache()

        # Load understanding model
        self.mammothmoda_und = self.MammothModa(
            pretrained=self.pretrained,
            mode="understanding",
            attn_implementation=self.attn_implementation,
            torch_dtype=self.torch_dtype_str,
            max_new_tokens=self.stage2_max_new_tokens,
            do_sample=self.stage2_do_sample,
            temperature=self.stage2_temperature,
            seed=self.seed,
            continual_mode=False,
        )

        self.current_model_mode = "understanding"
        eval_logger.info("Understanding model loaded successfully")

    @property
    def rank(self):
        # Return rank from whichever model is currently loaded
        if self.mammothmoda_gen is not None:
            return self.mammothmoda_gen.rank
        elif self.mammothmoda_und is not None:
            return self.mammothmoda_und.rank
        else:
            return 0

    @property
    def world_size(self):
        # Return world_size from whichever model is currently loaded
        if self.mammothmoda_gen is not None:
            return self.mammothmoda_gen.world_size
        elif self.mammothmoda_und is not None:
            return self.mammothmoda_und.world_size
        else:
            return 1

    @property
    def model(self):
        # Return whichever model is currently loaded
        if self.mammothmoda_gen is not None:
            return self.mammothmoda_gen.model
        elif self.mammothmoda_und is not None:
            return self.mammothmoda_und.model
        else:
            return None

    @property
    def tokenizer(self):
        # Return tokenizer from whichever model is currently loaded
        if self.mammothmoda_gen is not None:
            return self.mammothmoda_gen.tokenizer
        elif self.mammothmoda_und is not None:
            return self.mammothmoda_und.tokenizer
        else:
            return None

    def _stage1_generate_image(
        self, generation_prompt: str, doc_id: str, task: str, original_image=None
    ) -> Tuple[str, List[str]]:
        """
        Stage 1: Generate visualization image from prompt (conditioned on original image)

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
        if original_image is not None:
            eval_logger.debug("Stage 1 - Using original image as conditioning input")

        try:
            # Ensure generation model is loaded
            self._load_generation_model()

            text, images = self.mammothmoda_gen.generate_text_and_image(
                prompt=generation_prompt,
                doc_id=f"{doc_id}_stage1",
                task=task,
                image=original_image,
            )
            eval_logger.debug(f"Stage 1 - Generated {len(images)} image(s)")
            return text, images
        except Exception as e:
            eval_logger.error(f"Stage 1 failed for doc {doc_id}: {e}")
            if self.fail_gracefully:
                return "", []
            else:
                raise

    def _stage2_answer_with_image(
        self, question: str, image_path: str, doc_id: str, original_image: Optional[Image.Image] = None
    ) -> str:
        """
        Stage 2: Answer question using generated image (and optionally original image)

        Args:
            question: Original question text
            image_path: Path to generated auxiliary image
            doc_id: Document ID for logging
            original_image: Original image (optional, used as primary reference)

        Returns:
            Answer text
        """
        eval_logger.debug(f"Stage 2 - Answering question for doc {doc_id}")
        eval_logger.debug(f"Question: {question}")

        try:
            # Ensure understanding model is loaded
            self._load_understanding_model()

            # Load generated auxiliary image
            auxiliary_image = Image.open(image_path).convert("RGB")

            # If original image is provided, use both images
            # For MammothModa, we'll create a prompt that references both images
            if original_image is not None:
                eval_logger.debug("Stage 2 - Using both original and auxiliary images")
                # Create a prompt that instructs the model to use both images
                enhanced_question = f"Based on the provided images (original and auxiliary diagram), {question}"
                # Use the original image as primary input, auxiliary as context
                # Note: MammothModa's understand_image takes a single image
                # We'll use the original image as the main input
                answer = self.mammothmoda_und.understand_image(
                    prompt=enhanced_question,
                    image=original_image,
                    doc_id=f"{doc_id}_stage2",
                )
            else:
                eval_logger.debug("Stage 2 - Using only auxiliary image")
                # Use only the generated auxiliary image
                answer = self.mammothmoda_und.understand_image(
                    prompt=question,
                    image=auxiliary_image,
                    doc_id=f"{doc_id}_stage2",
                )

            eval_logger.debug(f"Stage 2 - Generated answer: {answer[:100]}...")
            return answer

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
        stage1_text: str,
        generated_images: List[str],
        question: str,
        stage2_answer: str,
    ):
        """Save intermediate artifacts for debugging and analysis"""
        if not self.save_intermediate:
            return

        # Create task-specific directory
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

    def generate_until(self, requests: List[Instance]) -> List[str]:
        """
        Main inference method implementing two-stage visual CoT with memory-efficient batch processing

        Pass 1: Generate all visualization images (using generation model)
        Pass 2: Answer all questions (using understanding model)

        This approach loads only one model at a time to save GPU memory.
        """
        import re

        eval_logger.info(f"\n{'='*60}")
        eval_logger.info(f"Starting two-pass inference for {len(requests)} requests")
        eval_logger.info(f"{'='*60}")

        # Store intermediate results for each request
        stage1_results = []

        # ========== PASS 1: Generate all auxiliary images ==========
        eval_logger.info("\n[PASS 1] Generating auxiliary images...")
        self._load_generation_model()

        pbar_stage1 = tqdm(
            total=len(requests),
            disable=(self.rank != 0),
            desc="Stage 1: Generating Images",
        )

        for request in requests:
            contexts, gen_kwargs, doc_to_visual, doc_id, task, split = request.args

            # Extract original image from document
            original_image = None
            if doc_to_visual is not None:
                try:
                    doc = self.task_dict[task][split][doc_id]
                    original_visuals = doc_to_visual(doc)
                    if original_visuals and len(original_visuals) > 0:
                        original_image = original_visuals[0]
                        eval_logger.debug(f"Extracted original image for doc {doc_id}")
                except Exception as e:
                    eval_logger.warning(f"Failed to extract original image for doc {doc_id}: {e}")

            # Parse contexts to extract generation_prompt if provided
            gen_prompt_match = re.search(r'\[GEN_PROMPT\](.*?)\[/GEN_PROMPT\]', contexts, re.DOTALL)
            question_match = re.search(r'\[QUESTION\](.*?)\[/QUESTION\]', contexts, re.DOTALL)

            if gen_prompt_match and question_match:
                custom_gen_prompt = gen_prompt_match.group(1).strip()
                actual_question = question_match.group(1).strip()
                generation_prompt = custom_gen_prompt.replace("{question}", actual_question)
                # Update contexts to be just the question for stage 2
                question_text = question_match.group(1)
            else:
                generation_prompt = self.generation_prompt_template.format(question=contexts)
                question_text = contexts

            # Stage 1: Generate visualization image
            stage1_text, generated_images = self._stage1_generate_image(
                generation_prompt=generation_prompt,
                doc_id=doc_id,
                task=task,
                original_image=original_image,
            )

            # Store results for Stage 2
            stage1_results.append({
                "doc_id": doc_id,
                "task": task,
                "question": question_text,
                "generation_prompt": generation_prompt,
                "stage1_text": stage1_text,
                "generated_images": generated_images,
                "original_image": original_image,
            })

            pbar_stage1.update(1)

        pbar_stage1.close()

        # ========== PASS 2: Answer all questions using generated images ==========
        eval_logger.info("\n[PASS 2] Answering questions with generated images...")
        self._load_understanding_model()

        res = []
        pbar_stage2 = tqdm(
            total=len(requests),
            disable=(self.rank != 0),
            desc="Stage 2: Answering Questions",
        )

        for stage1_result in stage1_results:
            doc_id = stage1_result["doc_id"]
            task = stage1_result["task"]
            question = stage1_result["question"]
            generation_prompt = stage1_result["generation_prompt"]
            stage1_text = stage1_result["stage1_text"]
            generated_images = stage1_result["generated_images"]
            original_image = stage1_result["original_image"]

            # Check if image was generated
            if not generated_images or len(generated_images) == 0:
                eval_logger.warning(
                    f"No image generated for doc {doc_id}, using stage 1 text as answer"
                )
                res.append(stage1_text if stage1_text else "")
                pbar_stage2.update(1)
                continue

            # Stage 2: Answer question using generated image
            final_answer = self._stage2_answer_with_image(
                question=question,
                image_path=generated_images[0],
                doc_id=doc_id,
                original_image=original_image
            )

            # Save intermediate artifacts if enabled
            self._save_intermediate_artifacts(
                doc_id=doc_id,
                task=task,
                generation_prompt=generation_prompt,
                stage1_text=stage1_text,
                generated_images=generated_images,
                question=question,
                stage2_answer=final_answer,
            )

            res.append(final_answer)
            pbar_stage2.update(1)

        pbar_stage2.close()

        eval_logger.info(f"\n{'='*60}")
        eval_logger.info(f"Completed two-pass inference for {len(requests)} requests")
        eval_logger.info(f"{'='*60}")

        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """Not implemented for visual CoT model"""
        raise NotImplementedError(
            "loglikelihood is not supported for MammothModaVisualCoT. "
            "This model is designed for generation tasks only."
        )

    def generate_until_multi_round(self, requests: List[Instance]) -> List[str]:
        """Not implemented yet"""
        raise NotImplementedError("generate_until_multi_round is not yet implemented for MammothModaVisualCoT")
