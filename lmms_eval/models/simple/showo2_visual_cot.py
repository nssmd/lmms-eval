# coding=utf-8
# Copyright 2025 NUS Show Lab.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Show-o2 Visual Chain-of-Thought Model

Two-stage inference:
1. Stage 1: Generate visualization image from text prompt
2. Stage 2: Answer question using the generated image

Usage:
    python -m lmms_eval \
        --model showo2_visual_cot \
        --model_args pretrained=showlab/show-o2-7B \
        --tasks mathvista_visual_cot \
        --batch_size 1 \
        --output_path ./logs/
"""

import json
import os
from typing import List, Optional, Tuple

from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model


@register_model("showo2_visual_cot")
class Showo2VisualCoT(lmms):
    """
    Show-o2 Visual Chain-of-Thought Model

    Performs two-stage visual reasoning:
    1. Generate visualization image from text prompt
    2. Answer question using the generated image
    """

    def __init__(
        self,
        pretrained: str = "showlab/show-o2-7B",
        # Stage 1: Image generation parameters
        stage1_guidance_scale: float = 5.0,
        stage1_num_inference_steps: int = 50,
        stage1_resolution: int = 432,
        stage1_use_image_conditioning: bool = True,  # Enable i2i by default
        # Stage 2: Visual understanding parameters
        stage2_max_new_tokens: int = 512,
        stage2_top_k: int = 1,
        stage2_temperature: float = 1.0,
        # Generation prompt template
        generation_prompt_template: str = "Generate a detailed visual diagram or illustration to help answer this question: {question}",
        # Output and debugging
        output_dir: Optional[str] = None,
        save_intermediate: bool = False,
        intermediate_dir: Optional[str] = None,
        # Error handling
        fail_gracefully: bool = True,
        # Model loading
        llm_model_path: str = "Qwen/Qwen2.5-7B-Instruct",
        vae_model_path: Optional[str] = None,
        weight_type: str = "bfloat16",
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
        self.stage1_guidance_scale = stage1_guidance_scale
        self.stage1_num_inference_steps = stage1_num_inference_steps
        self.stage1_resolution = stage1_resolution
        self.stage1_use_image_conditioning = stage1_use_image_conditioning

        # Stage 2 parameters
        self.stage2_max_new_tokens = stage2_max_new_tokens
        self.stage2_top_k = stage2_top_k
        self.stage2_temperature = stage2_temperature

        # Model loading parameters
        self.llm_model_path = llm_model_path
        self.vae_model_path = vae_model_path
        self.weight_type = weight_type

        # Setup output directories
        if output_dir is None:
            self.output_dir = "./logs/showo2_visual_cot"
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

        # Import and initialize Show-o2 model
        eval_logger.info(f"Loading Show-o2 model from {pretrained}")
        self._load_showo2_model()

        eval_logger.info("Showo2VisualCoT initialized successfully")

    def _load_showo2_model(self):
        """Initialize model loading parameters (actual loading is deferred)"""
        # Don't load models here - we'll load them on demand to save memory
        self.showo2 = None
        self.current_mode = None
        eval_logger.info("Show-o2 model loading deferred (will load on demand)")

    def _load_model_for_mode(self, mode: str):
        """Load model for specific mode, unloading previous model if needed"""
        import gc
        import torch
        from lmms_eval.models.simple.showo2 import Showo2

        if self.current_mode == mode and self.showo2 is not None:
            return  # Already loaded for this mode

        # Unload previous model if exists
        if self.showo2 is not None:
            eval_logger.info(f"Unloading Show-o2 model (was in {self.current_mode} mode)")
            del self.showo2
            self.showo2 = None
            gc.collect()
            torch.cuda.empty_cache()

        # Load model for new mode
        eval_logger.info(f"Loading Show-o2 model for {mode} mode")
        self.showo2 = Showo2(
            pretrained=self.pretrained,
            mode=mode,
            llm_model_path=self.llm_model_path,
            vae_model_path=self.vae_model_path,
            resolution=self.stage1_resolution,
            weight_type=self.weight_type,
            output_image_dir=self.intermediate_dir,
            guidance_scale=self.stage1_guidance_scale,
            num_inference_steps=self.stage1_num_inference_steps,
            max_new_tokens=self.stage2_max_new_tokens,
            top_k=self.stage2_top_k,
            temperature=self.stage2_temperature,
            seed=self.seed,
            continual_mode=False,
        )
        self.current_mode = mode
        eval_logger.info(f"Show-o2 model loaded for {mode} mode")

    @property
    def rank(self):
        return 0  # Default rank

    @property
    def world_size(self):
        return 1  # Default world size

    @property
    def model(self):
        return self.showo2.model if self.showo2 else None

    @property
    def tokenizer(self):
        return self.showo2.tokenizer if self.showo2 else None

    def _stage1_generate_image(
        self, generation_prompt: str, doc_id: str, task: str, original_image=None
    ) -> Tuple[str, List[str]]:
        """
        Stage 1: Generate visualization image from prompt (optionally conditioned on original image)

        Args:
            generation_prompt: Text prompt for image generation
            doc_id: Document ID for file naming
            task: Task name for file naming
            original_image: Original image to condition on (optional, used if stage1_use_image_conditioning=True)

        Returns:
            Tuple of (generated_text, list_of_image_paths)
        """
        # ========== STAGE 1 DEBUG INFO ==========
        eval_logger.info("=" * 100)
        eval_logger.info(f"🎨 STAGE 1 IMAGE GENERATION - Doc: {doc_id}")
        eval_logger.info("=" * 100)
        eval_logger.info(f"📋 Configuration:")
        eval_logger.info(f"   - stage1_use_image_conditioning: {self.stage1_use_image_conditioning}")
        eval_logger.info(f"   - original_image provided: {original_image is not None}")

        if original_image is not None:
            eval_logger.info(f"   - original_image size: {original_image.size}")
            eval_logger.info(f"   - original_image mode: {original_image.mode}")

        # Determine whether to use image conditioning
        use_image = original_image if self.stage1_use_image_conditioning else None

        if use_image is not None:
            eval_logger.info("")
            eval_logger.info("✅ MODE: IMAGE-TO-IMAGE (i2i) GENERATION")
            eval_logger.info("   → Stage1 will use BOTH image and text as input")
            eval_logger.info("   → Calling: generate_image_with_conditioning()")
            eval_logger.info("")
        else:
            eval_logger.info("")
            eval_logger.info("📝 MODE: TEXT-TO-IMAGE (t2i) GENERATION")
            if original_image is not None:
                eval_logger.info("   → Original image provided but stage1_use_image_conditioning=False")
            else:
                eval_logger.info("   → No original image provided")
            eval_logger.info("   → Stage1 will use ONLY text as input")
            eval_logger.info("   → Calling: generate_image()")
            eval_logger.info("")

        eval_logger.info(f"📝 Generation prompt: {generation_prompt[:150]}...")
        eval_logger.info("=" * 100)

        eval_logger.debug(f"Stage 1 - Generating image for doc {doc_id}")
        eval_logger.debug(f"Generation prompt: {generation_prompt}")

        try:
            # Load generation model
            self._load_model_for_mode("generation")

            if use_image is not None:
                # Use mixed modality generation (i2i)
                eval_logger.info("🚀 Starting i2i generation with image conditioning...")
                text, images = self.showo2.generate_image_with_conditioning(
                    prompt=generation_prompt,
                    conditioning_image=use_image,
                    doc_id=f"{doc_id}_stage1",
                    task=task,
                )
                eval_logger.info(f"✅ i2i generation completed: {len(images)} image(s) generated")
            else:
                # Use pure text-to-image generation
                eval_logger.info("🚀 Starting t2i generation (text-only)...")
                text, images = self.showo2.generate_image(
                    prompt=generation_prompt,
                    doc_id=f"{doc_id}_stage1",
                    task=task,
                )
                eval_logger.info(f"✅ t2i generation completed: {len(images)} image(s) generated")

            eval_logger.debug(f"Stage 1 - Generated {len(images)} image(s)")
            return text, images
        except Exception as e:
            import traceback
            eval_logger.error(f"❌ Stage 1 failed for doc {doc_id}: {e}")
            eval_logger.error(f"Full traceback:\n{traceback.format_exc()}")
            if self.fail_gracefully:
                return "", []
            else:
                raise

    def _stage2_answer_with_image(
        self,
        question: str,
        image_path: str,
        doc_id: str,
        original_image: Optional[Image.Image] = None,
    ) -> str:
        """
        Stage 2: Answer question using generated image (and optionally original image)

        Args:
            question: Original question text
            image_path: Path to generated auxiliary image
            doc_id: Document ID for logging
            original_image: Original image (optional, will be used together with auxiliary)

        Returns:
            Answer text
        """
        eval_logger.debug(f"Stage 2 - Answering question for doc {doc_id}")
        eval_logger.debug(f"Question: {question}")

        try:
            # Load understanding model (will unload generation model first)
            self._load_model_for_mode("understanding")

            # Load generated auxiliary image
            auxiliary_image = Image.open(image_path).convert("RGB")

            # If original image is provided, use both images
            if original_image is not None:
                eval_logger.debug("Stage 2 - Using both original and auxiliary images")
                # Update question to provide context about the two images
                question_with_context = (
                    "You are given two images. The first image is the original image, "
                    "and the second image is an auxiliary visualization to help answer the question. "
                    + question
                )
                answer_text = self.showo2.understand_two_images(
                    prompt=question_with_context,
                    image1=original_image,
                    image2=auxiliary_image,
                    doc_id=doc_id,
                )
            else:
                # Use only auxiliary image
                eval_logger.debug("Stage 2 - Using auxiliary image only")
                answer_text = self.showo2.understand_image(
                    prompt=question,
                    image=auxiliary_image,
                    doc_id=doc_id,
                )

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
        stage1_text: str,
        generated_images: List[str],
        question: str,
        stage2_answer: str,
    ) -> None:
        """Save intermediate artifacts for debugging"""
        if not self.save_intermediate:
            return

        artifact_dir = os.path.join(self.intermediate_dir, task)
        os.makedirs(artifact_dir, exist_ok=True)

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
        Main inference method implementing two-stage visual CoT

        Stage 1: Generate visualization image from text prompt
        Stage 2: Answer question using the generated image

        Also supports Uni-MMMU interleaved generation mode.
        """
        res = []
        pbar = tqdm(
            total=len(requests),
            disable=(self.rank != 0),
            desc="Showo2VisualCoT Generating",
        )

        for request in requests:
            contexts, gen_kwargs, doc_to_visual, doc_id, task, split = request.args

            # Check if this is Uni-MMMU interleaved generation mode
            showo2_interleaved = gen_kwargs.get("showo2_interleaved", None)
            if showo2_interleaved is None:
                # Also check for bagel_interleaved for compatibility
                showo2_interleaved = gen_kwargs.get("bagel_interleaved", None)

            if showo2_interleaved is not None:
                # Uni-MMMU interleaved generation mode
                eval_logger.info(f"Using Uni-MMMU interleaved mode for doc {doc_id}")

                # Get input images and doc data
                doc = self.task_dict[task][split][doc_id]
                input_images = []
                if doc_to_visual is not None:
                    visuals = [doc_to_visual(doc)]
                    input_images = self.flatten(visuals)

                output_text, output_images = self.generate_uni_mmmu_interleaved(
                    input_images, contexts, str(doc_id), task, showo2_interleaved, doc
                )
                formatted_output = self.format_output(output_text, output_images)
                res.append(formatted_output)
                pbar.update(1)
                continue

            # Standard two-stage visual CoT mode
            # Extract original image from document using task_dict
            original_image = None
            if doc_to_visual is not None:
                try:
                    doc = self.task_dict[task][split][doc_id]
                    original_visuals = doc_to_visual(doc)
                    if original_visuals and len(original_visuals) > 0:
                        original_image = original_visuals[0]
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
                custom_gen_prompt = gen_prompt_match.group(1).strip()
                actual_question = question_match.group(1).strip()
                generation_prompt = custom_gen_prompt.replace(
                    "{question}", actual_question
                )
                contexts = contexts.replace(
                    f"[GEN_PROMPT]{gen_prompt_match.group(1)}[/GEN_PROMPT]", ""
                )
                contexts = contexts.replace(
                    f"[QUESTION]{question_match.group(1)}[/QUESTION]",
                    question_match.group(1),
                )
                eval_logger.info("Using custom generation prompt from task config")
            else:
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

            # Check if image was generated
            if not generated_images or len(generated_images) == 0:
                eval_logger.warning(
                    f"No image generated for doc {doc_id}, using stage 1 text as answer"
                )
                res.append(stage1_text if stage1_text else "")
                pbar.update(1)
                continue

            # Stage 2: Answer question using generated image
            final_answer = self._stage2_answer_with_image(
                question=contexts,
                image_path=generated_images[0],
                doc_id=doc_id,
                original_image=original_image,
            )

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
            pbar.update(1)

        pbar.close()
        return res

    def flatten(self, input_list):
        """Flatten a nested list"""
        output = []
        for item in input_list:
            if isinstance(item, list):
                output.extend(self.flatten(item))
            else:
                output.append(item)
        return output

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """Not supported for generation models"""
        raise NotImplementedError(
            "Showo2VisualCoT is a generation model and does not support loglikelihood"
        )

    def generate_until_multi_round(self, requests) -> List[str]:
        """Multi-round dialogue not yet implemented"""
        raise NotImplementedError(
            "Multi-round dialogue not yet implemented for Showo2VisualCoT"
        )

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
        Uni-MMMU interleaved generation for Show-o2.

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

        # Override generation params if specified
        guidance_scale = interleaved_config.get("guidance_scale", self.stage1_guidance_scale)
        num_inference_steps = interleaved_config.get("num_inference_steps", self.stage1_num_inference_steps)

        # Text generation params
        text_max_new_tokens = interleaved_config.get("text_max_new_tokens", self.stage2_max_new_tokens)
        text_temperature = interleaved_config.get("text_temperature", self.stage2_temperature)

        generated_images = []

        # Load generation model for image generation
        self._load_model_for_mode("generation")

        if task_type == "jigsaw":
            # Jigsaw: Generate 2 completed images then final answer
            eval_logger.info("Uni-MMMU Jigsaw mode: generating 2 candidate images")

            # Generate Candidate 0 image
            suffix1 = "Output ONLY a single image with Candidate 0 placed in the bottom-right cell. No text."
            full_prompt_0 = prompt + "\n" + suffix1

            # Use first input image as conditioning
            conditioning_image = input_images[0] if input_images and len(input_images) > 0 else None

            img0_text, img0_paths = self._generate_image_interleaved(
                prompt=full_prompt_0,
                conditioning_image=conditioning_image,
                doc_id=f"{doc_id}_cand0",
                task=task,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
            )
            if img0_paths and len(img0_paths) > 0:
                generated_images.append(img0_paths[0])
                img0 = Image.open(img0_paths[0]).convert("RGB")
                eval_logger.info(f"Generated Candidate 0: {img0_paths[0]}")
            else:
                eval_logger.error("Failed to generate Candidate 0")
                return "", []

            # Generate Candidate 1 image (use img0 as additional conditioning)
            suffix2 = "Output ONLY a single image with Candidate 1 placed in the bottom-right cell. No text."
            full_prompt_1 = prompt + "\nCOMPLETED WITH CANDIDATE 0:\n" + suffix2

            img1_text, img1_paths = self._generate_image_interleaved(
                prompt=full_prompt_1,
                conditioning_image=img0,  # Use generated img0 as conditioning
                doc_id=f"{doc_id}_cand1",
                task=task,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
            )
            if img1_paths and len(img1_paths) > 0:
                generated_images.append(img1_paths[0])
                img1 = Image.open(img1_paths[0]).convert("RGB")
                eval_logger.info(f"Generated Candidate 1: {img1_paths[0]}")
            else:
                eval_logger.error("Failed to generate Candidate 1")
                return "", generated_images

            # Generate final answer using understanding model
            self._load_model_for_mode("understanding")

            final_suffix = (
                'Now output EXACTLY ONE <FINAL_ANSWER_JSON>{"choice": 0 or 1, "rationale": "≤30 words"}</FINAL_ANSWER_JSON>\n'
                "Do not output any additional images."
            )

            # Build context with all images
            final_prompt = (
                "You are given multiple images:\n"
                "- Original input images (reference puzzle and candidate patches)\n"
                "- Generated Candidate 0 completion\n"
                "- Generated Candidate 1 completion\n\n"
                + prompt + "\n\n"
                "COMPLETED WITH CANDIDATE 0:\n"
                "COMPLETED WITH CANDIDATE 1:\n\n"
                + final_suffix
            )

            # Use understand_two_images with the two generated candidates
            final_text = self.showo2.understand_two_images(
                prompt=final_prompt,
                image1=img0,
                image2=img1,
                doc_id=f"{doc_id}_final",
            )

        else:
            # Maze/Sliding: [gen_text(plan) → gen_image(step)]×k → gen_text(answer)
            eval_logger.info(f"Uni-MMMU {task_type} mode: generating {num_images} steps")

            step_texts = []  # Store all plan texts
            step_images = []  # Store all generated step images

            # Start with understanding model for text generation
            self._load_model_for_mode("understanding")

            # Use first input image as base context
            base_image = input_images[0] if input_images and len(input_images) > 0 else None

            for i in range(1, num_images + 1):
                # Generate planning text
                if task_type == "maze":
                    plan_suffix = f'Now planning for step {i}, Please output a sentence in the form: "Next, move one step up/down/left/right."'
                else:  # sliding
                    plan_suffix = f'Now planning for step {i}, Please output a sentence describing which tile to move and in which direction.'

                plan_prompt = prompt + "\n" + "\n".join(step_texts) + "\n" + plan_suffix

                if base_image is not None:
                    plan_text = self.showo2.understand_image(
                        prompt=plan_prompt,
                        image=base_image,
                        doc_id=f"{doc_id}_plan_{i}",
                    )
                else:
                    # Fallback: use text-only (though this shouldn't happen in uni_mmmu)
                    plan_text = f"Step {i} plan"

                eval_logger.info(f"Step {i} plan: {plan_text}")
                step_texts.append(plan_text)

                # Generate step image
                self._load_model_for_mode("generation")

                img_suffix = f"Now, generate the image for step {i}."
                img_prompt = prompt + "\n" + "\n".join(step_texts) + "\n" + img_suffix

                # Use previous step image as conditioning if available, otherwise use base image
                conditioning_img = step_images[-1] if step_images else base_image

                step_img_text, step_img_paths = self._generate_image_interleaved(
                    prompt=img_prompt,
                    conditioning_image=conditioning_img,
                    doc_id=f"{doc_id}_step_{i:04d}",
                    task=task,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                )

                if step_img_paths and len(step_img_paths) > 0:
                    generated_images.append(step_img_paths[0])
                    step_img = Image.open(step_img_paths[0]).convert("RGB")
                    step_images.append(step_img)
                    eval_logger.info(f"Generated step {i} image: {step_img_paths[0]}")
                else:
                    eval_logger.error(f"Failed to generate step {i} image")
                    # Continue anyway
                    step_images.append(base_image)

                # Switch back to understanding model for next iteration
                self._load_model_for_mode("understanding")

            # Generate final answer
            final_suffix = (
                "After the images, emit EXACTLY ONE LINE containing ONLY the final move list "
                "as <ANSWER_JSON>[...]</ANSWER_JSON>. No other text."
            )

            final_prompt = (
                prompt + "\n\n"
                + "\n".join([f"Step {i+1}: {text}" for i, text in enumerate(step_texts)])
                + "\n\n" + final_suffix
            )

            # Use the last generated image for final answer
            if step_images:
                final_text = self.showo2.understand_image(
                    prompt=final_prompt,
                    image=step_images[-1],
                    doc_id=f"{doc_id}_final",
                )
            else:
                final_text = ""

            eval_logger.info(f"{task_type} final answer: {final_text}")

        return final_text, generated_images

    def _generate_image_interleaved(
        self,
        prompt: str,
        conditioning_image: Optional[Image.Image],
        doc_id: str,
        task: str,
        guidance_scale: float,
        num_inference_steps: int,
    ) -> Tuple[str, List[str]]:
        """
        Helper method to generate image for interleaved generation.

        Args:
            prompt: Text prompt for image generation
            conditioning_image: Optional conditioning image for i2i
            doc_id: Document ID for file naming
            task: Task name for file naming
            guidance_scale: Guidance scale for generation
            num_inference_steps: Number of inference steps

        Returns:
            Tuple of (generated_text, list_of_image_paths)
        """
        try:
            # Temporarily override generation parameters
            original_guidance = self.showo2.guidance_scale
            original_steps = self.showo2.num_inference_steps

            self.showo2.guidance_scale = guidance_scale
            self.showo2.num_inference_steps = num_inference_steps

            if conditioning_image is not None:
                # Use i2i generation
                text, images = self.showo2.generate_image_with_conditioning(
                    prompt=prompt,
                    conditioning_image=conditioning_image,
                    doc_id=doc_id,
                    task=task,
                )
            else:
                # Use t2i generation
                text, images = self.showo2.generate_image(
                    prompt=prompt,
                    doc_id=doc_id,
                    task=task,
                )

            # Restore original parameters
            self.showo2.guidance_scale = original_guidance
            self.showo2.num_inference_steps = original_steps

            return text, images

        except Exception as e:
            eval_logger.error(f"Image generation failed for {doc_id}: {e}")
            if self.fail_gracefully:
                return "", []
            else:
                raise

    def format_output(self, text: str, images: List[str]) -> str:
        """Format output as JSON string"""
        eval_logger.debug(f"[FORMAT OUTPUT] Input: text type={type(text).__name__}, text value={repr(text)}, images count={len(images) if images else 0}")
        output_dict = {"text": text, "images": images}
        result = json.dumps(output_dict, ensure_ascii=False)
        eval_logger.debug(f"[FORMAT OUTPUT] Output JSON: {result[:200]}..." if len(result) > 200 else f"[FORMAT OUTPUT] Output JSON: {result}")
        return result
