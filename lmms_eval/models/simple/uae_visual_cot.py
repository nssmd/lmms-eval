"""
UAE Visual Chain-of-Thought Model

Two-stage inference:
1. Stage 1: Generate auxiliary visualization image from original image + text prompt
2. Stage 2: Answer question using original image + auxiliary image + question

Usage:
    python -m lmms_eval \
        --model uae_visual_cot \
        --model_args pretrained=zhiyuanyan1/UAE \
        --tasks illusionbench_arshia_logo_shape_visual_cot \
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


@register_model("uae_visual_cot")
class UAEVisualCoT(lmms):
    """
    UAE Visual Chain-of-Thought Model

    Performs two-stage visual reasoning:
    1. Generate auxiliary visualization image from original image + text prompt
    2. Answer question using original image + auxiliary image + question
    """

    def __init__(
        self,
        pretrained: str = "zhiyuanyan1/UAE",
        # Stage 1: Image generation parameters
        stage1_sd3_num_inference_steps: int = 40,
        stage1_sd3_guidance_scale: float = 5.0,
        stage1_sd3_height: int = 1024,
        stage1_sd3_width: int = 1024,
        stage1_use_original_image: bool = True,
        # Stage 2: Visual understanding parameters
        stage2_llm_max_new_tokens: int = 512,
        stage2_llm_temperature: float = 0.0,
        stage2_llm_do_sample: bool = False,
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
        seed: int = 42,
        **kwargs,
    ) -> None:
        super().__init__()

        self.pretrained = pretrained
        self.save_intermediate = save_intermediate
        self.fail_gracefully = fail_gracefully
        self.seed = seed
        self.generation_prompt_template = generation_prompt_template
        self.stage1_use_original_image = stage1_use_original_image

        # Stage 1 parameters
        self.stage1_sd3_num_inference_steps = stage1_sd3_num_inference_steps
        self.stage1_sd3_guidance_scale = stage1_sd3_guidance_scale
        self.stage1_sd3_height = stage1_sd3_height
        self.stage1_sd3_width = stage1_sd3_width

        # Stage 2 parameters
        self.stage2_llm_max_new_tokens = stage2_llm_max_new_tokens
        self.stage2_llm_temperature = stage2_llm_temperature
        self.stage2_llm_do_sample = stage2_llm_do_sample

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

        # Import and initialize UAE model
        eval_logger.info(f"Loading UAE model from {pretrained}")
        self._load_uae_model(device)

        eval_logger.info("UAEVisualCoT initialized successfully")

    def _load_uae_model(self, device: str):
        """Load UAE model with both generation and understanding capabilities"""
        from lmms_eval.models.simple.uae import UAE

        # Initialize UAE with generation mode for Stage 1
        # We'll use the same model for both stages
        self.uae = UAE(
            pretrained=self.pretrained,
            mode="generation",  # Need generation for Stage 1
            device=device,
            sd3_num_inference_steps=self.stage1_sd3_num_inference_steps,
            sd3_guidance_scale=self.stage1_sd3_guidance_scale,
            sd3_height=self.stage1_sd3_height,
            sd3_width=self.stage1_sd3_width,
            llm_max_new_tokens=self.stage2_llm_max_new_tokens,
            llm_temperature=self.stage2_llm_temperature,
            llm_do_sample=self.stage2_llm_do_sample,
            output_image_dir=self.intermediate_dir,
            seed=self.seed,
            continual_mode=False,  # Disable caching for visual CoT
        )

        eval_logger.info("UAE model loaded successfully")

    @property
    def rank(self):
        return self.uae.rank

    @property
    def world_size(self):
        return self.uae.world_size

    @property
    def model(self):
        return self.uae.model

    @property
    def tokenizer(self):
        return self.uae.tokenizer

    def _stage1_generate_image(
        self,
        generation_prompt: str,
        doc_id: str,
        task: str,
        original_image: Optional[Image.Image] = None,
    ) -> Tuple[str, List[str]]:
        """
        Stage 1: Generate auxiliary visualization image from prompt
        (conditioned on original image if provided)

        Args:
            generation_prompt: Text prompt for image generation
            doc_id: Document ID for file naming
            task: Task name for file naming
            original_image: Original image to condition on (required for proper visual CoT)

        Returns:
            Tuple of (generated_text, list_of_image_paths)
        """
        eval_logger.debug(f"Stage 1 - Generating image for doc {doc_id}")
        eval_logger.debug(f"Generation prompt: {generation_prompt}")
        if original_image is not None:
            eval_logger.debug("Stage 1 - Using original image as conditioning input")

        try:
            # Use original image if stage1_use_original_image is True
            input_image = original_image if self.stage1_use_original_image else None

            text, images = self.uae.generate_image(
                prompt=generation_prompt,
                doc_id=f"{doc_id}_stage1",
                task=task,
                image=input_image,
            )
            eval_logger.debug(f"Stage 1 - Generated {len(images)} image(s)")
            return text, images
        except Exception as e:
            eval_logger.error(f"Stage 1 failed for doc {doc_id}: {e}")
            if self.fail_gracefully:
                return "", []
            else:
                raise

    def _stage2_answer_with_images(
        self,
        question: str,
        auxiliary_image_path: str,
        doc_id: str,
        original_image: Optional[Image.Image] = None,
    ) -> str:
        """
        Stage 2: Answer question using original image + auxiliary image

        Args:
            question: Original question text
            auxiliary_image_path: Path to generated auxiliary image
            doc_id: Document ID for logging
            original_image: Original image (required for proper visual CoT)

        Returns:
            Answer text
        """
        eval_logger.debug(f"Stage 2 - Answering question for doc {doc_id}")
        eval_logger.debug(f"Question: {question}")

        try:
            # Load auxiliary image
            auxiliary_image = Image.open(auxiliary_image_path).convert("RGB")

            # Prepare multi-image input
            if original_image is not None:
                eval_logger.debug(
                    "Stage 2 - Using both original and auxiliary images"
                )
                # Create a combined prompt with both images
                # Qwen-2.5-VL supports multiple images in messages
                try:
                    from qwen_vl_utils import process_vision_info
                except ImportError:
                    eval_logger.warning(
                        "qwen_vl_utils not found, using basic image processing"
                    )
                    process_vision_info = None

                # Prepare messages with two images
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

                # Apply chat template
                text = self.uae._processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

                # Process vision info
                if process_vision_info is not None:
                    image_inputs, video_inputs = process_vision_info(messages)
                else:
                    image_inputs = [original_image, auxiliary_image]
                    video_inputs = None

                # Prepare model inputs
                model_inputs = self.uae._processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    return_tensors="pt",
                    padding=True,
                ).to(self.uae._device)

                # Generate
                import torch

                with torch.no_grad():
                    generated_ids = self.uae._encoder.generate(
                        **model_inputs,
                        max_new_tokens=self.stage2_llm_max_new_tokens,
                        do_sample=self.stage2_llm_do_sample,
                        temperature=self.stage2_llm_temperature,
                    )

                # Decode
                generated_ids = [
                    output_ids[len(input_ids) :]
                    for input_ids, output_ids in zip(
                        model_inputs.input_ids, generated_ids
                    )
                ]
                answer_text = self.uae._processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )[0]

            else:
                # Fallback to auxiliary image only
                eval_logger.debug("Stage 2 - Using auxiliary image only")
                answer_text = self.uae.understand_image(
                    question, auxiliary_image, doc_id
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
        Main inference method implementing two-stage visual CoT

        Stage 1: Generate auxiliary visualization image from original image + text prompt
        Stage 2: Answer question using original image + auxiliary image
        """
        res = []
        pbar = tqdm(
            total=len(requests),
            disable=(self.rank != 0),
            desc="UAEVisualCoT Generating",
        )

        for request in requests:
            contexts, gen_kwargs, doc_to_visual, doc_id, task, split = request.args

            # Extract original image from document using task_dict
            original_image = None
            if doc_to_visual is not None:
                try:
                    # Get doc from task_dict
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

            # Stage 1: Generate auxiliary visualization image (with original image as input)
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

            # Stage 2: Answer question using original image + auxiliary image
            final_answer = self._stage2_answer_with_images(
                question=contexts,
                auxiliary_image_path=generated_images[0],
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

            # Return only final answer text
            res.append(final_answer)
            pbar.update(1)

        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """Not supported for generation models"""
        raise NotImplementedError(
            "UAEVisualCoT is a generation model and does not support loglikelihood"
        )

    def generate_until_multi_round(self, requests) -> List[str]:
        """Multi-round dialogue not yet implemented"""
        raise NotImplementedError(
            "Multi-round dialogue not yet implemented for UAEVisualCoT"
        )
