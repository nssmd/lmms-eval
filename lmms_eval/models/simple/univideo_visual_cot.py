"""
UniVideo Visual Chain-of-Thought Model

Two-stage inference:
1. Stage 1: Generate auxiliary visualization image from original image + text prompt
2. Stage 2: Answer question using original image + auxiliary image + text prompt

Usage:
    python -m lmms_eval \
        --model univideo_visual_cot \
        --model_args pretrained=/path/to/UniVideo,mllm_device=cuda:0,diffusion_device=cuda:1 \
        --tasks mathvista_visual_cot \
        --batch_size 1 \
        --output_path ./logs/
"""

import json
import os
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model


@register_model("univideo_visual_cot")
class UniVideoVisualCoT(lmms):
    """
    UniVideo Visual Chain-of-Thought Model

    Performs two-stage visual reasoning:
    1. Stage 1: Generate auxiliary visualization image from original image + text prompt
    2. Stage 2: Answer question using original image + auxiliary image + text prompt

    Multi-GPU Support:
        - GPU 0 (cuda:0): MLLM encoder (Qwen2.5-VL-7B)
        - GPU 1 (cuda:1): VAE + Transformer (HunyuanVideo)
    """

    def __init__(
        self,
        pretrained: str,
        # Stage 1: Image generation parameters
        stage1_num_inference_steps: int = 50,
        stage1_guidance_scale: float = 7.0,
        stage1_image_guidance_scale: float = 2.0,
        stage1_timestep_shift: float = 7.0,
        stage1_height: int = 480,
        stage1_width: int = 832,
        # Stage 2: Visual understanding parameters
        stage2_max_new_tokens: int = 1000,
        # Generation prompt template
        generation_prompt_template: str = "Generate a detailed visual diagram or illustration to help answer this question: {question}",
        # Output and debugging
        output_dir: Optional[str] = None,
        save_intermediate: bool = False,
        intermediate_dir: Optional[str] = None,
        # Error handling
        fail_gracefully: bool = True,
        # Model loading
        config_path: Optional[str] = None,
        mllm_device: str = "cuda:0",
        diffusion_device: str = "cuda:1",
        seed: int = 42,
        continual_mode: bool = False,
        response_persistent_folder: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        self.pretrained = pretrained
        self.save_intermediate = save_intermediate
        self.fail_gracefully = fail_gracefully
        self.seed = seed
        self.generation_prompt_template = generation_prompt_template
        self.continual_mode = continual_mode

        # Stage 1 parameters (image generation)
        self.stage1_num_inference_steps = stage1_num_inference_steps
        self.stage1_guidance_scale = stage1_guidance_scale
        self.stage1_image_guidance_scale = stage1_image_guidance_scale
        self.stage1_timestep_shift = stage1_timestep_shift
        self.stage1_height = stage1_height
        self.stage1_width = stage1_width

        # Stage 2 parameters (understanding)
        self.stage2_max_new_tokens = stage2_max_new_tokens

        # Multi-GPU device configuration
        self.mllm_device = torch.device(mllm_device)
        self.diffusion_device = torch.device(diffusion_device)

        # Setup output directories
        if output_dir is None:
            self.output_dir = "./logs/univideo_visual_cot"
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

        # Setup response cache for continual mode
        self.response_cache = {}
        self.cache_mode = "start"

        if response_persistent_folder is None:
            self.response_persistent_folder = "./logs/univideo_visual_cot_persistent"
        else:
            self.response_persistent_folder = response_persistent_folder

        if self.continual_mode:
            os.makedirs(self.response_persistent_folder, exist_ok=True)
            self.response_persistent_file = os.path.join(
                self.response_persistent_folder, "univideo_visual_cot_response.json"
            )

            if os.path.exists(self.response_persistent_file):
                with open(self.response_persistent_file, "r") as f:
                    self.response_cache = json.load(f)
                self.cache_mode = "resume"
                eval_logger.info(f"Loaded cache: {len(self.response_cache)} records")

        # Import and initialize UniVideo model
        eval_logger.info(f"Loading UniVideo model from {pretrained}")
        self._load_univideo_model(config_path)

        eval_logger.info("UniVideoVisualCoT initialized successfully")

    def _load_univideo_model(self, config_path: Optional[str]):
        """Load UniVideo model with both generation and understanding capabilities"""
        from lmms_eval.models.simple.univideo import UniVideo

        # Initialize UniVideo with both generation and understanding capabilities
        self.univideo = UniVideo(
            pretrained=self.pretrained,
            mode="understanding",  # Will switch modes as needed
            config_path=config_path,
            output_image_dir=self.intermediate_dir,
            max_new_tokens=self.stage2_max_new_tokens,
            num_inference_steps=self.stage1_num_inference_steps,
            guidance_scale=self.stage1_guidance_scale,
            image_guidance_scale=self.stage1_image_guidance_scale,
            timestep_shift=self.stage1_timestep_shift,
            height=self.stage1_height,
            width=self.stage1_width,
            num_frames=1,  # Single image generation
            seed=self.seed,
            mllm_device=str(self.mllm_device),
            diffusion_device=str(self.diffusion_device),
            continual_mode=False,  # Disable caching for visual CoT
        )

        eval_logger.info("UniVideo model loaded successfully")

    @property
    def rank(self):
        return self.univideo.rank

    @property
    def world_size(self):
        return self.univideo.world_size

    @property
    def model(self):
        return self.univideo.model

    @property
    def tokenizer(self):
        return self.univideo.tokenizer

    def _stage1_generate_auxiliary_image(
        self,
        generation_prompt: str,
        original_image: Image.Image,
        doc_id: str,
        task: str,
    ) -> Tuple[str, List[str]]:
        """
        Stage 1: Generate auxiliary visualization image from original image + prompt

        Uses UniVideo's i2i_edit task to generate an auxiliary image conditioned on
        the original image and the generation prompt.

        Args:
            generation_prompt: Text prompt for image generation
            original_image: Original image to condition on
            doc_id: Document ID for file naming
            task: Task name for file naming

        Returns:
            Tuple of (generated_text, list_of_image_paths)
        """
        eval_logger.debug(f"Stage 1 - Generating auxiliary image for doc {doc_id}")
        eval_logger.debug(f"Generation prompt: {generation_prompt}")

        try:
            self.univideo.set_seed(self.seed)

            # Save original image to temp file for pipeline
            # Convert to RGB to handle RGBA images (JPEG doesn't support alpha channel)
            rgb_image = original_image.convert("RGB")
            with tempfile.NamedTemporaryFile(
                suffix=".jpg", delete=False
            ) as tmp:
                rgb_image.save(tmp.name)
                cond_image_path = tmp.name

            # Negative prompt for generation
            negative_prompt = (
                "Bright tones, overexposed, oversharpening, static, blurred details, "
                "subtitles, style, works, paintings, images, static, overall gray, "
                "worst quality, low quality, JPEG compression residue, ugly, incomplete, "
                "extra fingers, poorly drawn hands, poorly drawn faces, deformed, "
                "disfigured, misshapen limbs, fused fingers, still picture, messy background"
            )

            # Prepare pipeline kwargs for i2i_edit task
            pipeline_kwargs = {
                "prompts": [generation_prompt],
                "negative_prompt": negative_prompt,
                "cond_image_path": cond_image_path,
                "height": self.stage1_height,
                "width": self.stage1_width,
                "num_frames": 1,
                "num_inference_steps": self.stage1_num_inference_steps,
                "guidance_scale": self.stage1_guidance_scale,
                "image_guidance_scale": self.stage1_image_guidance_scale,
                "seed": self.seed,
                "timestep_shift": self.stage1_timestep_shift,
                "task": "i2i_edit",  # Use image-to-image editing
            }

            # Run pipeline
            output = self.univideo.pipeline(**pipeline_kwargs)

            # Save output image
            output_images = []
            if hasattr(output, "frames") and output.frames is not None:
                frames = output.frames[0]  # (F, H, W, C)
                if hasattr(frames, "detach"):
                    frames = frames.detach().cpu().float().numpy()

                F, H, W, C = frames.shape
                if F >= 1:
                    img = frames[0]
                    if img.min() < 0:
                        img = (img + 1.0) / 2.0
                    img = (img * 255).clip(0, 255).astype(np.uint8)

                    # Create task-specific directory
                    task_dir = os.path.join(self.intermediate_dir, task)
                    os.makedirs(task_dir, exist_ok=True)

                    safe_filename = f"{doc_id}_stage1_auxiliary.png"
                    image_path = os.path.join(task_dir, safe_filename)
                    Image.fromarray(img).save(image_path)
                    output_images.append(image_path)
                    eval_logger.info(f"Stage 1 - Saved auxiliary image: {image_path}")

            # Clean up temp file
            try:
                os.unlink(cond_image_path)
            except Exception:
                pass

            eval_logger.debug(f"Stage 1 - Generated {len(output_images)} image(s)")
            return "", output_images

        except Exception as e:
            eval_logger.error(f"Stage 1 failed for doc {doc_id}: {e}")
            if self.fail_gracefully:
                return "", []
            else:
                raise

    def _stage2_answer_with_images(
        self,
        question: str,
        original_image: Image.Image,
        auxiliary_image_path: str,
        doc_id: str,
    ) -> str:
        """
        Stage 2: Answer question using original image + auxiliary image + prompt

        Uses UniVideo's understanding mode with both images as input.

        Args:
            question: Original question text
            original_image: Original image
            auxiliary_image_path: Path to generated auxiliary image
            doc_id: Document ID for logging

        Returns:
            Answer text
        """
        eval_logger.debug(f"Stage 2 - Answering question for doc {doc_id}")
        eval_logger.debug(f"Question: {question}")

        try:
            self.univideo.set_seed(self.seed)

            # Load auxiliary image
            auxiliary_image = Image.open(auxiliary_image_path).convert("RGB")

            # Convert original image to RGB to handle RGBA images
            original_image_rgb = original_image.convert("RGB")

            # Prepare prompt that references both images
            # The prompt instructs the model to use both the original and auxiliary images
            combined_prompt = (
                f"You are given two images:\n"
                f"1. The original image (first image)\n"
                f"2. An auxiliary visualization image (second image) that may help answer the question\n\n"
                f"Please analyze both images carefully and answer the following question:\n{question}"
            )

            # Use MLLM for understanding with both images
            tokenize_fn = self.univideo.pipeline.mllm_encoder.get_tokenize_fn()
            tokenizer = self.univideo.pipeline.mllm_encoder.get_tokenizer()

            # Prepare images list - both original and auxiliary
            images = [[original_image_rgb, auxiliary_image]]

            batch = tokenize_fn(
                tokenizer, [combined_prompt], images, None, add_queires=False
            )

            # Move inputs to MLLM device
            inputs = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(self.mllm_device)
                else:
                    inputs[k] = v

            # Run generation on MLLM
            output_text = self.univideo.pipeline.mllm_encoder.generation(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                pixel_values=inputs.get("pixel_values"),
                image_grid_thw=inputs.get("image_grid_thw"),
                pixel_values_videos=inputs.get("pixel_values_videos"),
                video_grid_thw=inputs.get("video_grid_thw"),
                second_per_grid_ts=inputs.get("second_per_grid_ts"),
            )

            answer_text = output_text[0] if output_text else ""
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

    def flatten(self, input_list):
        """Flatten a nested list"""
        output = []
        for item in input_list:
            if isinstance(item, list):
                output.extend(self.flatten(item))
            else:
                output.append(item)
        return output

    def generate_until(self, requests: List[Instance]) -> List[str]:
        """
        Main inference method implementing two-stage visual CoT

        Stage 1: Generate auxiliary visualization image from original image + text prompt
        Stage 2: Answer question using original image + auxiliary image + text prompt
        """
        res = []
        pbar = tqdm(
            total=len(requests),
            disable=(self.rank != 0),
            desc="UniVideoVisualCoT Generating",
        )

        def get_uuid(task, split, doc_id):
            return f"{task}___{split}___{doc_id}"

        for request in requests:
            contexts, gen_kwargs, doc_to_visual, doc_id, task, split = request.args
            doc_uuid = get_uuid(task, split, doc_id)

            # Check cache
            if self.continual_mode and self.cache_mode == "resume":
                if doc_uuid in self.response_cache:
                    content = self.response_cache[doc_uuid]
                    if content:
                        res.append(content)
                        pbar.update(1)
                        continue

            # Extract original image from document using task_dict
            original_image = None
            if doc_to_visual is not None:
                try:
                    doc = self.task_dict[task][split][doc_id]
                    original_visuals = doc_to_visual(doc)
                    if original_visuals and len(original_visuals) > 0:
                        visual = original_visuals[0]
                        if isinstance(visual, Image.Image):
                            original_image = visual
                        elif isinstance(visual, str):
                            original_image = Image.open(visual).convert("RGB")
                        eval_logger.debug(
                            f"Extracted original image for doc {doc_id}"
                        )
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
                contexts = actual_question
                eval_logger.info("Using custom generation prompt from task config")
            else:
                # Use default template
                generation_prompt = self.generation_prompt_template.format(
                    question=contexts
                )

            eval_logger.info(f"\n{'='*60}")
            eval_logger.info(f"Processing doc {doc_id} from task {task}")
            eval_logger.info(f"{'='*60}")

            # Check if we have an original image
            if original_image is None:
                eval_logger.warning(
                    f"No original image for doc {doc_id}, falling back to text-only"
                )
                # Fallback to text-only understanding
                try:
                    output_text = self.univideo.understand_visual(
                        prompt=contexts, image=None, video=None, doc_id=str(doc_id)
                    )
                    res.append(output_text)
                except Exception as e:
                    eval_logger.error(f"Text-only fallback failed: {e}")
                    res.append("")
                pbar.update(1)
                continue

            # Stage 1: Generate auxiliary visualization image (with original image as input)
            _, generated_images = self._stage1_generate_auxiliary_image(
                generation_prompt=generation_prompt,
                original_image=original_image,
                doc_id=str(doc_id),
                task=task,
            )

            # Check if auxiliary image was generated
            if not generated_images or len(generated_images) == 0:
                eval_logger.warning(
                    f"No auxiliary image generated for doc {doc_id}, "
                    f"falling back to single-image understanding"
                )
                # Fallback to single image understanding
                try:
                    output_text = self.univideo.understand_visual(
                        prompt=contexts,
                        image=original_image,
                        video=None,
                        doc_id=str(doc_id),
                    )
                    res.append(output_text)
                except Exception as e:
                    eval_logger.error(f"Single-image fallback failed: {e}")
                    res.append("")
                pbar.update(1)
                continue

            # Stage 2: Answer question using original image + auxiliary image
            final_answer = self._stage2_answer_with_images(
                question=contexts,
                original_image=original_image,
                auxiliary_image_path=generated_images[0],
                doc_id=str(doc_id),
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

            # Update cache
            if self.continual_mode:
                self.response_cache[doc_uuid] = final_answer
                with open(self.response_persistent_file, "w") as f:
                    json.dump(self.response_cache, f, ensure_ascii=False, indent=2)

            pbar.update(1)

        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """Not supported for generation models"""
        raise NotImplementedError(
            "UniVideoVisualCoT is a generation model and does not support loglikelihood"
        )

    def generate_until_multi_round(self, requests) -> List[str]:
        """Multi-round dialogue not yet implemented"""
        raise NotImplementedError(
            "Multi-round dialogue not yet implemented for UniVideoVisualCoT"
        )
