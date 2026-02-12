"""
UniWorld Visual Chain-of-Thought Model

Two-stage inference:
1. Stage 1: Generate visualization image using UniWorld's generation mode
2. Stage 2: Answer question using UniWorld's understanding mode (Qwen2.5-VL)
"""

import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from PIL import Image
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval import utils

eval_logger = utils.eval_logger


@register_model("uniworld_visual_cot")
class UniWorldVisualCoT(lmms):
    """
    UniWorld Visual Chain-of-Thought Model

    Performs two-stage visual reasoning:
    1. Generate visualization using UniWorld's generation pipeline
    2. Answer question using Qwen2.5-VL understanding
    """

    def __init__(
        self,
        pretrained: str = "LanguageBind/UniWorld-V1",
        flux_path: str = "black-forest-labs/FLUX.1-dev",
        siglip_path: str = "google/siglip2-so400m-patch16-512",
        # Stage 1: Image generation parameters
        stage1_height: int = 1024,
        stage1_width: int = 1024,
        stage1_num_inference_steps: int = 28,
        stage1_guidance_scale: float = 3.5,
        # Stage 2: Visual understanding parameters
        stage2_max_new_tokens: int = 512,
        stage2_temperature: float = 0.0,
        stage2_do_sample: bool = False,
        # Output and debugging
        output_dir: Optional[str] = None,
        save_intermediate: bool = True,
        # Hugging Face upload
        hf_repo: Optional[str] = None,
        hf_upload: bool = False,
        # Model loading
        min_pixels: int = 448 * 448,
        max_pixels: int = 448 * 448,
        no_joint_with_t5: bool = False,
        offload: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()

        self.pretrained = pretrained
        self.flux_path = flux_path
        self.siglip_path = siglip_path
        self.save_intermediate = save_intermediate

        # Stage 1 parameters
        self.stage1_height = stage1_height
        self.stage1_width = stage1_width
        self.stage1_num_inference_steps = stage1_num_inference_steps
        self.stage1_guidance_scale = stage1_guidance_scale

        # Stage 2 parameters
        self.stage2_max_new_tokens = stage2_max_new_tokens
        self.stage2_temperature = stage2_temperature
        self.stage2_do_sample = stage2_do_sample

        # UniWorld parameters
        # Handle string format like "112*112" from command line
        if isinstance(min_pixels, str):
            self.min_pixels = eval(min_pixels) if '*' in min_pixels else int(min_pixels)
        else:
            self.min_pixels = min_pixels

        if isinstance(max_pixels, str):
            self.max_pixels = eval(max_pixels) if '*' in max_pixels else int(max_pixels)
        else:
            self.max_pixels = max_pixels

        self.no_joint_with_t5 = no_joint_with_t5
        self.offload = offload

        # Hugging Face upload
        self.hf_upload = hf_upload
        self.hf_repo = hf_repo
        self.hf_api = None

        if self.hf_upload:
            if not self.hf_repo:
                eval_logger.warning("hf_upload=True but hf_repo not specified, disabling upload")
                self.hf_upload = False
            else:
                try:
                    from huggingface_hub import HfApi
                    self.hf_api = HfApi()
                    eval_logger.info(f"✅ Hugging Face upload enabled: {self.hf_repo}")
                except ImportError:
                    eval_logger.warning("huggingface_hub not installed, disabling upload")
                    self.hf_upload = False

        # Setup output directories
        if output_dir is None:
            self.output_dir = "./logs/uniworld_visual_cot"
        else:
            self.output_dir = output_dir

        os.makedirs(self.output_dir, exist_ok=True)
        eval_logger.info(f"Output directory: {self.output_dir}")

        # Load UniWorld models
        eval_logger.info(f"Loading UniWorld model from {pretrained}")
        self._load_uniworld_models()

        eval_logger.info("UniWorldVisualCoT initialized successfully")

    def _load_uniworld_models(self):
        """Load UniWorld model with full capabilities (generation + understanding)"""
        from lmms_eval.models.simple.uniworld import UniWorld

        # Load UniWorld with generation mode (includes all models: Qwen2.5-VL + FLUX + SigLIP)
        eval_logger.info("Loading UniWorld with full capabilities...")
        self.uniworld = UniWorld(
            pretrained=self.pretrained,
            flux_path=self.flux_path,
            siglip_path=self.siglip_path,
            mode="generation",  # Load all models
            height=self.stage1_height,
            width=self.stage1_width,
            num_inference_steps=self.stage1_num_inference_steps,
            guidance_scale=self.stage1_guidance_scale,
            max_new_tokens=self.stage2_max_new_tokens,
            do_sample=self.stage2_do_sample,
            temperature=self.stage2_temperature,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
            no_joint_with_t5=self.no_joint_with_t5,
            offload=self.offload,
            image_output_dir=self.output_dir,
        )

        eval_logger.info("UniWorld loaded successfully (generation + understanding)")

    @property
    def rank(self):
        return self.uniworld.rank if hasattr(self.uniworld, 'rank') else 0

    @property
    def world_size(self):
        return self.uniworld.world_size if hasattr(self.uniworld, 'world_size') else 1

    @property
    def model(self):
        return self.uniworld.model

    @property
    def batch_size(self):
        return 1  # Visual CoT processes one at a time

    def generate_until(self, requests: List[Instance]) -> List[str]:
        """Main inference method implementing two-stage visual CoT"""
        res = []
        pbar = tqdm(
            total=len(requests),
            disable=(self.rank != 0),
            desc="UniWorldVisualCoT",
        )

        for request in requests:
            contexts, gen_kwargs, doc_to_visual, doc_id, task, split = request.args

            try:
                # Check for Uni-MMMU interleaved mode
                # Support both uniworld_interleaved and bagel_interleaved for compatibility
                uniworld_interleaved = gen_kwargs.get("uniworld_interleaved", None) or gen_kwargs.get("bagel_interleaved", None)

                if uniworld_interleaved is not None:
                    # Uni-MMMU interleaved generation mode
                    if not hasattr(self, 'task_dict'):
                        raise ValueError(f"task_dict not set for model")

                    doc = self.task_dict[task][split][doc_id]
                    input_images = []
                    if doc_to_visual:
                        visuals = [doc_to_visual(doc)]
                        input_images = self.flatten(visuals)

                    output_text, output_images = self.generate_uni_mmmu_interleaved(
                        input_images, contexts, str(doc_id), task, uniworld_interleaved, doc
                    )
                    output = json.dumps({"text": output_text, "images": output_images}, ensure_ascii=False)
                    res.append(output)
                    pbar.update(1)
                    continue

                # Standard two-stage Visual CoT mode
                # Extract original image (required for Visual CoT)
                if doc_to_visual is None:
                    raise ValueError(f"doc_to_visual is None for doc {doc_id}")

                if not hasattr(self, 'task_dict'):
                    raise ValueError(f"task_dict not set for model")

                if task not in self.task_dict:
                    raise ValueError(f"Task '{task}' not found in task_dict. Available: {list(self.task_dict.keys())}")

                doc = self.task_dict[task][split][doc_id]
                visuals = [doc_to_visual(doc)]
                original_images = self.flatten(visuals)

                if not original_images:
                    raise ValueError(f"No original images found for doc {doc_id}")

                eval_logger.info(f"[Doc {doc_id}] Loaded {len(original_images)} original image(s)")

                # Stage 1: Generate visualization
                eval_logger.info(f"[Doc {doc_id}] Stage 1: Generating visualization...")
                generated_image_path = self._stage1_generate(
                    prompt=contexts,
                    doc_id=doc_id,
                    task=task,
                    original_images=original_images,
                )

                if not generated_image_path:
                    eval_logger.warning(f"No image generated for doc {doc_id}")
                    res.append("")
                    pbar.update(1)
                    continue

                # Stage 2: Answer with generated image
                eval_logger.info(f"[Doc {doc_id}] Stage 2: Understanding with visualization...")
                final_answer = self._stage2_understand(
                    prompt=contexts,
                    generated_image_path=generated_image_path,
                    original_images=original_images,
                    doc_id=doc_id,
                )

                res.append(final_answer)
                eval_logger.info(f"[Doc {doc_id}] ✅ Answer: {final_answer[:100]}...")

                # Clear GPU cache to prevent OOM
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                eval_logger.error(f"Error in visual CoT for doc_id={doc_id}: {e}")
                import traceback
                traceback.print_exc()
                res.append("")

            pbar.update(1)

        pbar.close()
        return res

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
        Uni-MMMU interleaved generation aligned with Bagel's implementation.

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
        task_type = interleaved_config.get("task_type", "jigsaw")
        num_images = interleaved_config.get("num_images", 2)

        # Get num_images dynamically from doc if available
        if doc is not None:
            if task_type == "maze":
                steps_str = doc.get("steps", "[]")
                steps = json.loads(steps_str) if isinstance(steps_str, str) else steps_str
                if steps:
                    num_images = len(steps)
            elif task_type == "sliding":
                steps_str = doc.get("steps_words", "[]")
                steps = json.loads(steps_str) if isinstance(steps_str, str) else steps_str
                if steps:
                    num_images = len(steps)

        generated_images = []

        if task_type == "jigsaw":
            eval_logger.info(f"[Doc {doc_id}] Jigsaw mode: generating 2 candidate completions")

            # Generate Candidate 0 completion
            suffix1 = "Output ONLY a single image with Candidate 0 placed in the bottom-right cell. No text."
            full_prompt1 = f"{prompt}\n{suffix1}"

            eval_logger.info(f"[Doc {doc_id}] Generating Candidate 0 completion...")
            img0_path = self._generate_single_image(
                prompt=full_prompt1,
                input_images=input_images,
                doc_id=f"{doc_id}_cand0",
                task=task
            )
            if img0_path:
                generated_images.append(img0_path)
                eval_logger.info(f"[Doc {doc_id}] ✅ Candidate 0: {img0_path}")

            # Generate Candidate 1 completion
            suffix2 = "Output ONLY a single image with Candidate 1 placed in the bottom-right cell. No text."
            full_prompt2 = f"{prompt}\n{suffix2}"

            eval_logger.info(f"[Doc {doc_id}] Generating Candidate 1 completion...")
            img1_path = self._generate_single_image(
                prompt=full_prompt2,
                input_images=input_images,
                doc_id=f"{doc_id}_cand1",
                task=task
            )
            if img1_path:
                generated_images.append(img1_path)
                eval_logger.info(f"[Doc {doc_id}] ✅ Candidate 1: {img1_path}")

            # Generate final answer using all images
            # Align with Bagel: original images → prompt → cand0 image → text → cand1 image → text → final_suffix
            final_suffix = (
                'Now output EXACTLY ONE <FINAL_ANSWER_JSON>{"choice": 0 or 1, "rationale": "≤30 words"}</FINAL_ANSWER_JSON>\n'
                "Do not output any additional images."
            )
            final_question = f"{prompt}\n\nCOMPLETED WITH CANDIDATE 0:\n\nCOMPLETED WITH CANDIDATE 1:\n\n{final_suffix}"

            # Prepare all images for final answer
            all_images = input_images + generated_images
            eval_logger.info(f"[Doc {doc_id}] Generating final answer with {len(all_images)} images...")
            final_text = self._generate_text_answer(
                question=final_question,
                images=all_images,
                doc_id=doc_id
            )

            return final_text, generated_images

        else:
            # Maze/Sliding: [gen_text(plan) → gen_image(step)]×k → gen_text(answer)
            eval_logger.info(f"[Doc {doc_id}] {task_type.capitalize()} mode: generating {num_images} steps")

            step_texts = []
            accumulated_images = list(input_images)  # Start with input images

            for i in range(1, num_images + 1):
                # Generate planning text
                if task_type == "maze":
                    plan_suffix = f'Now planning for step {i}, Please output a sentence in the form: "Next, move one step up/down/left/right."'
                else:  # sliding
                    plan_suffix = f'Now planning for step {i}, Please output a sentence describing which tile to move and in which direction.'

                # Build context with all previous steps
                context_text = prompt
                for j, prev_text in enumerate(step_texts, 1):
                    context_text += f"\n\nStep {j} plan: {prev_text}"
                context_text += f"\n\n{plan_suffix}"

                eval_logger.info(f"[Doc {doc_id}] Generating step {i} plan...")
                plan_text = self._generate_text_answer(
                    question=context_text,
                    images=accumulated_images,
                    doc_id=f"{doc_id}_plan_{i}",
                    max_tokens=128
                )
                step_texts.append(plan_text)
                eval_logger.info(f"[Doc {doc_id}] Step {i} plan: {plan_text}")

                # Generate step image
                img_suffix = f"Now, generate the image for step {i}."
                img_prompt = f"{context_text}\n\n{plan_text}\n\n{img_suffix}"

                eval_logger.info(f"[Doc {doc_id}] Generating step {i} image...")
                img_path = self._generate_single_image(
                    prompt=img_prompt,
                    input_images=accumulated_images,
                    doc_id=f"{doc_id}_step_{i:04d}",
                    task=task
                )

                if img_path:
                    generated_images.append(img_path)
                    accumulated_images.append(img_path)
                    eval_logger.info(f"[Doc {doc_id}] ✅ Step {i} image: {img_path}")

            # Generate final answer with all steps
            # Align with Bagel: original images → prompt → [plan_text → "Image for step i:" → step_image]×k → final_suffix
            final_suffix = (
                "After the images, emit EXACTLY ONE LINE containing ONLY the final move list "
                "as <ANSWER_JSON>[...]</ANSWER_JSON>. No other text."
            )

            # Build final context with all steps (aligned with Bagel)
            final_context = prompt
            for i, plan_text in enumerate(step_texts, 1):
                final_context += f"\n\n{plan_text}"
                final_context += f"\nImage for step {i}:"
            final_context += f"\n\n{final_suffix}"

            eval_logger.info(f"[Doc {doc_id}] Generating final answer with {len(accumulated_images)} images...")
            final_text = self._generate_text_answer(
                question=final_context,
                images=accumulated_images,
                doc_id=doc_id
            )

            return final_text, generated_images

    def _generate_single_image(
        self,
        prompt: str,
        input_images: List,
        doc_id: str,
        task: str
    ) -> Optional[str]:
        """
        Generate a single image using UniWorld's generation mode.

        Args:
            prompt: Text prompt for image generation
            input_images: List of input images (PIL Images or paths)
            doc_id: Document ID for file naming
            task: Task name for file naming

        Returns:
            Path to generated image, or None if generation failed
        """
        from qwen_vl_utils import process_vision_info

        try:
            # Prepare messages for image generation
            messages = [
                {
                    "role": "user",
                    "content": [
                        *[{"type": "image", "image": img} for img in input_images if img is not None],
                        {"type": "text", "text": prompt}
                    ]
                }
            ]

            # Process inputs
            text = self.uniworld.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = self.uniworld.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.uniworld._device)

            # Call UniWorld's image generation method
            image_output = self.uniworld._generate_image(
                inputs=inputs,
                prompt_text=prompt,
                history_image_paths=[],
                doc_id=doc_id,
                task=task,
            )

            # Parse output to get image path
            if isinstance(image_output, str) and image_output.startswith("{"):
                output_dict = json.loads(image_output)
                images = output_dict.get("images", [])
                if images:
                    return images[0]

            return None

        except Exception as e:
            eval_logger.error(f"Error generating image for doc_id={doc_id}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _generate_text_answer(
        self,
        question: str,
        images: List,
        doc_id: str,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate text answer using UniWorld's understanding mode.

        Args:
            question: Question text
            images: List of images (PIL Images or paths)
            doc_id: Document ID for logging
            max_tokens: Maximum tokens to generate (default: use stage2_max_new_tokens)

        Returns:
            Generated text answer
        """
        from qwen_vl_utils import process_vision_info

        try:
            # Prepare conversation with multiple images
            content = [{"type": "text", "text": question}]
            for img in images:
                if img is not None:
                    content.append({
                        "type": "image",
                        "image": img,
                        "min_pixels": self.min_pixels,
                        "max_pixels": self.max_pixels
                    })

            messages = [{"role": "user", "content": content}]

            # Process inputs
            text = self.uniworld.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = self.uniworld.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.uniworld._device)

            # Generate text
            with torch.no_grad():
                outputs = self.uniworld.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens if max_tokens else self.stage2_max_new_tokens,
                    do_sample=self.stage2_do_sample,
                    temperature=self.stage2_temperature if self.stage2_do_sample else None,
                    pad_token_id=self.uniworld.processor.tokenizer.pad_token_id,
                    eos_token_id=self.uniworld.processor.tokenizer.eos_token_id,
                )

            # Decode output
            generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
            answer = self.uniworld.processor.tokenizer.decode(
                generated_ids, skip_special_tokens=True
            ).strip()

            return answer

        except Exception as e:
            eval_logger.error(f"Error generating text answer for doc_id={doc_id}: {e}")
            import traceback
            traceback.print_exc()
            return ""

    def _stage1_generate(
        self,
        prompt: str,
        doc_id: int,
        task: str,
        original_images: List,
    ) -> Optional[str]:
        """Stage 1: Generate visualization image"""
        from qwen_vl_utils import process_vision_info
        import re

        # Extract generation prompt from [GEN_PROMPT]...[/GEN_PROMPT] tags
        gen_match = re.search(r'\[GEN_PROMPT\](.*?)\[/GEN_PROMPT\]', prompt, re.DOTALL)
        if gen_match:
            gen_prompt = gen_match.group(1).strip()
        else:
            # Fallback: use the whole prompt with a generation instruction
            gen_prompt = f"{prompt}\n\nGenerate a clear schematic visualization to help understand this problem."

        try:
            # Prepare messages for image generation
            messages = [
                {
                    "role": "user",
                    "content": [
                        *[{"type": "image", "image": img} for img in original_images],
                        {"type": "text", "text": gen_prompt}
                    ]
                }
            ]

            # Process inputs
            text = self.uniworld.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = self.uniworld.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.uniworld._device)

            # Call UniWorld's image generation method directly
            image_path = self.uniworld._generate_image(
                prompt_text=gen_prompt,
                inputs=inputs,
                history_image_paths=[],
                doc_id=doc_id,
                task=task,
            )

            # Parse output to get image path
            if isinstance(image_path, str) and image_path.startswith("{"):
                output_dict = json.loads(image_path)
                images = output_dict.get("images", [])
                if images:
                    actual_path = images[0]
                    # Upload to HF if enabled
                    if self.hf_upload and self.hf_api:
                        self._upload_to_hf(actual_path, f"logs/{task}/images")
                    return actual_path

            return None

        except Exception as e:
            eval_logger.error(f"Stage 1 generation failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _upload_to_hf(self, file_path: str, hf_path: str):
        """Upload file to Hugging Face"""
        try:
            self.hf_api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=f"{hf_path}/{os.path.basename(file_path)}",
                repo_id=self.hf_repo,
                repo_type="dataset",
            )
            eval_logger.debug(f"📤 Uploaded to HF: {file_path}")
        except Exception as e:
            eval_logger.warning(f"Failed to upload to HF: {e}")

    def _stage2_understand(
        self,
        prompt: str,
        generated_image_path: str,
        original_images: List,
        doc_id: int,
    ) -> str:
        """Stage 2: Understand with generated visualization"""
        import re

        try:
            # Extract question from [QUESTION]...[/QUESTION] tags
            question_match = re.search(r'\[QUESTION\](.*?)\[/QUESTION\]', prompt, re.DOTALL)
            if question_match:
                und_prompt = question_match.group(1).strip()
            else:
                # Fallback: use the whole prompt
                und_prompt = f"{prompt}\n\nBased on the visualization, provide your answer."

            # Load generated image
            generated_image = Image.open(generated_image_path).convert("RGB")

            # Combine original + generated images
            all_images = original_images + [generated_image]

            # Prepare messages for Qwen2.5-VL
            messages = [
                {
                    "role": "user",
                    "content": [
                        *[{"type": "image", "image": img} for img in all_images],
                        {"type": "text", "text": und_prompt}
                    ]
                }
            ]

            # Process with Qwen2.5-VL (use UniWorld's processor and model)
            from qwen_vl_utils import process_vision_info

            text = self.uniworld.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = self.uniworld.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.uniworld._device)

            # Generate response
            with torch.no_grad():
                generated_ids = self.uniworld.model.generate(
                    **inputs,
                    max_new_tokens=self.stage2_max_new_tokens,
                    do_sample=self.stage2_do_sample,
                    temperature=self.stage2_temperature if self.stage2_do_sample else None,
                )

            # Decode output
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.uniworld.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]

            return output_text.strip()

        except Exception as e:
            eval_logger.error(f"Stage 2 understanding failed: {e}")
            return ""

    def flatten(self, item):
        """Flatten nested lists"""
        if isinstance(item, list):
            output = []
            for sub_item in item:
                output.extend(self.flatten(sub_item))
            return output
        else:
            return [item]

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """Not supported"""
        raise NotImplementedError("UniWorldVisualCoT does not support loglikelihood")

    def generate_until_multi_round(self, requests) -> List[str]:
        """Not yet implemented"""
        raise NotImplementedError("Multi-round not yet implemented for UniWorldVisualCoT")
