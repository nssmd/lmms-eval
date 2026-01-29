"""
MIO Visual Chain-of-Thought Model

Two-stage inference:
1. Stage 1: Generate visualization image from text prompt (REQUIRES original image input)
2. Stage 2: Answer question using the generated image

Usage:
    python -m lmms_eval \
        --model mio_cot \
        --model_args pretrained=m-a-p/MIO-7B-Instruct \
        --tasks mathvista_visual_cot \
        --batch_size 1 \
        --device cuda:0 \
        --output_path ./logs/

"""

import json
import os
import sys
import re
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model


@register_model("mio_cot")
class MIOVisualCoT(lmms):
    """
    MIO Visual Chain-of-Thought Model

    Performs two-stage visual reasoning:
    1. Generate visualization image from text prompt (REQUIRES original image as conditioning)
    2. Answer question using the generated image
    """

    def __init__(
        self,
        pretrained: str = "m-a-p/MIO-7B-Instruct",
        # Stage 1: Image generation parameters
        stage1_cfg_scale: float = 5.0,
        stage1_num_inference_steps: int = 50,
        stage1_guidance_scale: float = 7.5,
        stage1_image_size: int = 512,
        # Stage 2: Visual understanding parameters
        stage2_max_new_tokens: int = 512,
        stage2_temperature: float = 0.0,
        stage2_do_sample: bool = False,
        stage2_num_beams: int = 1,
        stage2_top_p: float = 0.9,
        stage2_repetition_penalty: float = 1.0,
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
        dtype: str = "float16",
        seed: int = 0,
        **kwargs,
    ) -> None:
        super().__init__()

        self.pretrained = pretrained
        self.save_intermediate = save_intermediate
        self.fail_gracefully = fail_gracefully
        self.seed = seed
        self.generation_prompt_template = generation_prompt_template
        self._device = torch.device(device)
        self._dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype

        # Stage 1 parameters
        self.stage1_cfg_scale = stage1_cfg_scale
        self.stage1_num_inference_steps = stage1_num_inference_steps
        self.stage1_guidance_scale = stage1_guidance_scale
        self.stage1_image_size = stage1_image_size

        # Stage 2 parameters
        self.stage2_max_new_tokens = stage2_max_new_tokens
        self.stage2_temperature = stage2_temperature
        self.stage2_do_sample = stage2_do_sample
        self.stage2_num_beams = stage2_num_beams
        self.stage2_top_p = stage2_top_p
        self.stage2_repetition_penalty = stage2_repetition_penalty

        # Setup output directories
        if output_dir is None:
            self.output_dir = "./logs/mio_visual_cot"
        else:
            self.output_dir = output_dir

        if intermediate_dir is None:
            # Structure: {output_dir}/{task_name}/
            # No need to add model name again since output_dir already contains it
            self.intermediate_dir = self.output_dir
        else:
            self.intermediate_dir = intermediate_dir

        if save_intermediate:
            os.makedirs(self.intermediate_dir, exist_ok=True)
            eval_logger.info(f"Intermediate artifacts will be saved under: {self.intermediate_dir}")

        # Import and initialize MIO model
        eval_logger.info(f"Loading MIO model from {pretrained}")
        self._load_mio_model()

        eval_logger.info("MIOVisualCoT initialized successfully")

    def _load_mio_model(self):
        """Load MIO model with both generation and understanding capabilities"""
        # Check if MIO code is available
        mio_path = Path(__file__).parent.parent.parent.parent / "MIO"
        if not mio_path.exists():
            raise FileNotFoundError(
                f"MIO repository not found at {mio_path}. "
                "Please clone it: git clone https://github.com/MIO-Team/MIO.git"
            )
        
        # Add MIO to path
        if str(mio_path) not in sys.path:
            sys.path.insert(0, str(mio_path))
        
        try:
            from tokenization_mio import MIOTokenizer
            from transformers import AutoModelForCausalLM
        except ImportError as e:
            raise ImportError(
                f"Failed to import MIO dependencies: {e}\n"
                "Please install requirements: pip install -r MIO/requirements.txt"
            )
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.pretrained,
            torch_dtype=self._dtype,
            device_map="auto",
        ).eval()
        
        # Load tokenizer
        self.tokenizer = MIOTokenizer(self.pretrained, str(self._device))
        
        eval_logger.info("✅ MIO model loaded successfully")

    @property
    def rank(self):
        # For single GPU, return 0
        return 0

    @property
    def world_size(self):
        # For single GPU, return 1
        return 1

    @property
    def batch_size(self):
        return 1

    def flatten(self, input_list):
        """Flatten a nested list"""
        return [item for sublist in input_list for item in sublist]

    def _stage1_generate_image(
        self, generation_prompt: str, doc_id: str, task: str, original_image
    ) -> Tuple[str, List[str]]:
        """
        Stage 1: Generate visualization image from prompt (REQUIRES original image as conditioning)

        Args:
            generation_prompt: Text prompt for image generation
            doc_id: Document ID for file naming
            task: Task name for file naming
            original_image: Original image to condition on (REQUIRED, cannot be None)

        Returns:
            Tuple of (generated_text, list_of_image_paths)
        """
        eval_logger.info(f"Stage 1 - Generating visualization for doc {doc_id}")
        eval_logger.info(f"Stage 1 - Generation prompt: {generation_prompt[:100]}...")
        
        # CRITICAL: Original image is required for generation
        if original_image is None:
            error_msg = f"Stage 1 requires original image as input, but got None for doc {doc_id}"
            eval_logger.error(error_msg)
            if self.fail_gracefully:
                return "", []
            else:
                raise ValueError(error_msg)
        
        eval_logger.info(f"Stage 1 - ✅ Using original image as conditioning input")

        try:
            # Prepare original image path
            image_paths = []
            temp_files = []
            
            if isinstance(original_image, str):
                image_paths.append(original_image)
                eval_logger.debug(f"Stage 1 - Using image path: {original_image}")
            elif isinstance(original_image, Image.Image):
                # Save PIL image to temp file
                import tempfile
                tmp = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
                original_image.save(tmp.name, 'JPEG')
                tmp.close()
                image_paths.append(tmp.name)
                temp_files.append(tmp.name)
                eval_logger.debug(f"Stage 1 - Saved PIL image to temp file: {tmp.name}")
            else:
                error_msg = f"Stage 1 - Unsupported image type: {type(original_image)}"
                eval_logger.error(error_msg)
                raise ValueError(error_msg)

            # Prepare conversation for conditional image generation (image edit/img2img)
            # Based on MIO official implementation:
            # - Pure generation (imagen): no placeholder, batch_image_paths=None
            # - Understanding (img_und): has placeholder, batch_image_paths provided  
            # - Conditional generation (edit): placeholder + batch_image_paths, request new image in prompt
            # 
            # For Visual CoT: provide original image, ask model to generate visualization
            # The model will see the original image and generate NEW image tokens
            # 
            # CRITICAL: Force image generation by:
            # 1. Explicitly request in prompt
            # 2. Pre-fill assistant response with <image> token to trigger generation
            image_placeholder = "<image_placeholder_0>"
            prompt_with_image = f"{image_placeholder}\n{generation_prompt}\n\nPlease generate the visualization image now."
            
            eval_logger.info(f"Stage 1 - Generation prompt: {generation_prompt[:200]}...")
            eval_logger.info(f"Stage 1 - Conditional generation: providing original image as context")
            eval_logger.info(f"Stage 1 - ⚡ Pre-filling assistant response with <image> to force generation")

            conversations = [[{"role": "user", "content": prompt_with_image}]]

            # Apply chat template WITH input images (conditional image generation)
            # Set add_generation_prompt=False so we can manually add the trigger
            inputs_without_assistant = self.tokenizer.apply_chat_template(
                conversations,
                batch_image_paths=[image_paths],  # Provide original image for conditioning
                batch_speech_paths=None,
                mode='std',
                padding=False,  # Don't pad yet, we'll add trigger first
                truncation=True,
                max_length=2048,
                return_tensors='pt',
                add_generation_prompt=False  # We'll add it manually with <image> trigger
            )
            
            # Manually add assistant starter with <image> trigger
            # Format: <|im_start|>assistant\n<image>
            assistant_starter_with_trigger = "<|im_start|>assistant\n<image>"
            
            # Get the text sequence and append trigger
            text_sequence = self.tokenizer.tokenizer.decode(inputs_without_assistant['input_ids'][0], skip_special_tokens=False)
            text_sequence_with_trigger = text_sequence + "\n" + assistant_starter_with_trigger
            
            # Re-tokenize with trigger
            inputs = self.tokenizer.tokenizer(
                text_sequence_with_trigger,
                padding=True,
                truncation=True,
                max_length=2048,
                return_tensors='pt'
            )

            input_ids = inputs['input_ids'].to(self._device)
            attention_mask = inputs['attention_mask'].to(self._device)

            # Generate using official multimodal interleaved generation config
            # Per MIO official docs: num_beams=1, do_sample=True, repetition_penalty=1.15
            gen_config = {
                "num_beams": 1,  # Multimodal interleaved: num_beams=1
                "do_sample": True,  # Multimodal interleaved: do_sample=True
                "temperature": 1.0,
                "top_p": 0.7,
                "repetition_penalty": 1.15,
                "max_new_tokens": 512,
                "length_penalty": 1.0,
                "pad_token_id": self.tokenizer.tokenizer.pad_token_id,
                "eos_token_id": 7,  # <|im_end|>
            }

            with torch.inference_mode():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **gen_config
                )

            # Setup output directory for generated images
            stage1_output_dir = os.path.join(self.intermediate_dir, task, "stage1_generated")
            os.makedirs(stage1_output_dir, exist_ok=True)

            # Decode and extract generated images
            generated_sequences, generated_images, _ = self.tokenizer.detokenize(
                outputs,
                output_image_dir=stage1_output_dir,
                output_speech_dir=None,
                extract_assistant=True,
                save_images=True,  # Save generated images
                save_speeches=False
            )

            # Get paths of generated images
            generated_image_paths = []
            if generated_images and len(generated_images) > 0:
                for img_list in generated_images:
                    if img_list:  # Check if list is not empty
                        generated_image_paths.extend(img_list)

            eval_logger.info(f"Stage 1 - Generated {len(generated_image_paths)} image(s) for doc {doc_id}")
            
            # Clean up temp files
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                except:
                    pass

            generated_text = generated_sequences[0].strip() if generated_sequences else ""
            return generated_text, generated_image_paths

        except Exception as e:
            eval_logger.error(f"Stage 1 failed for doc {doc_id}: {e}")
            import traceback
            eval_logger.error(traceback.format_exc())
            if self.fail_gracefully:
                return "", []
            else:
                raise

    def _stage2_answer_with_image(
        self, question: str, image_path: Optional[str], doc_id: str, original_image: Optional[Image.Image] = None
    ) -> str:
        """
        Stage 2: Answer question using images (original and/or auxiliary)

        Args:
            question: Original question text (cleaned, without GEN_PROMPT tags)
            image_path: Path to generated auxiliary image (None if not generated)
            doc_id: Document ID for logging
            original_image: Original image (required if image_path is None)

        Returns:
            Answer text
        """
        eval_logger.debug(f"Stage 2 - Answering question for doc {doc_id}")
        eval_logger.debug(f"Question: {question}")

        try:
            # Prepare image paths: original image (if provided) + auxiliary image (if generated)
            image_paths = []
            temp_files = []

            # Add original image first (primary reference)
            if original_image is not None:
                if isinstance(original_image, str):
                    image_paths.append(original_image)
                elif isinstance(original_image, Image.Image):
                    import tempfile
                    tmp = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
                    original_image.save(tmp.name, 'JPEG')
                    tmp.close()
                    image_paths.append(tmp.name)
                    temp_files.append(tmp.name)
            
            # Add generated auxiliary image if available
            if image_path is not None:
                auxiliary_image = Image.open(image_path).convert("RGB")
                image_paths.append(image_path)
                eval_logger.info(f"Stage 2 - Using both original and auxiliary images for doc {doc_id}")
            else:
                eval_logger.info(f"Stage 2 - Using original image only for doc {doc_id}")
            
            if len(image_paths) == 0:
                raise ValueError(f"No images available for Stage 2 (doc {doc_id})")
                
            eval_logger.info(f"Stage 2 - Total images: {len(image_paths)} (order: {'original+auxiliary' if len(image_paths) > 1 else 'original only'})")

            # Prepare conversation with image placeholders
            image_placeholders = "".join([f"<image_placeholder_{i}>" for i in range(len(image_paths))])
            prompt_with_images = f"{image_placeholders}\n{question}"

            conversations = [[{"role": "user", "content": prompt_with_images}]]

            # Apply chat template for understanding mode
            inputs = self.tokenizer.apply_chat_template(
                conversations,
                batch_image_paths=[image_paths],
                batch_speech_paths=None,
                mode='std',  # Standard understanding mode
                padding=True,
                truncation=True,
                max_length=2048,
                return_tensors='pt'
            )

            input_ids = inputs['input_ids'].to(self._device)
            attention_mask = inputs['attention_mask'].to(self._device)

            # Generate answer
            gen_config = {
                "num_beams": self.stage2_num_beams,
                "do_sample": self.stage2_do_sample,
                "temperature": self.stage2_temperature,
                "top_p": self.stage2_top_p,
                "repetition_penalty": self.stage2_repetition_penalty,
                "max_new_tokens": self.stage2_max_new_tokens,
                "pad_token_id": self.tokenizer.tokenizer.pad_token_id,
                "eos_token_id": 7,  # <|im_end|>
            }

            with torch.inference_mode():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **gen_config
                )

            # Decode response
            generated_sequences, _, _ = self.tokenizer.detokenize(
                outputs,
                output_image_dir=None,
                output_speech_dir=None,
                extract_assistant=True,
                save_images=False,
                save_speeches=False
            )

            answer_text = generated_sequences[0].strip() if generated_sequences else ""

            # Clean up any residual tokens
            answer_text = re.sub(r'<img\d+>', '', answer_text)
            answer_text = re.sub(r'</?image>', '', answer_text)
            answer_text = re.sub(r'<spch\d+>', '', answer_text)
            answer_text = re.sub(r'</?spch>', '', answer_text)
            answer_text = answer_text.strip()

            eval_logger.debug(f"Stage 2 - Generated answer: {answer_text[:100]}...")
            
            # Clean up temp files
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                except:
                    pass

            return answer_text

        except Exception as e:
            eval_logger.error(f"Stage 2 failed for doc {doc_id}: {e}")
            import traceback
            eval_logger.error(traceback.format_exc())
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

        Stage 1: Generate visualization image from text prompt (with original image as input)
        Stage 2: Answer question using the generated image
        """
        res = []
        pbar = tqdm(
            total=len(requests),
            disable=(self.rank != 0),
            desc="MIOVisualCoT Generating",
        )

        for request in requests:
            contexts, gen_kwargs, doc_to_visual, doc_id, task, split = request.args

            # Extract original image from document using task_dict
            # CRITICAL: Original image is REQUIRED for visual CoT
            original_image = None
            if doc_to_visual is not None:
                try:
                    # Get doc from task_dict
                    doc = self.task_dict[task][split][doc_id]
                    original_visuals = doc_to_visual(doc)
                    if original_visuals and len(original_visuals) > 0:
                        original_image = original_visuals[0]
                        eval_logger.info(f"✅ Extracted original image for doc {doc_id}")
                    else:
                        eval_logger.error(f"❌ No original image found for doc {doc_id}, visual CoT requires original image!")
                except Exception as e:
                    eval_logger.error(f"❌ Failed to extract original image for doc {doc_id}: {e}")
            else:
                eval_logger.error(f"❌ doc_to_visual is None for doc {doc_id}, cannot extract original image!")

            # Parse contexts to extract generation_prompt if provided
            gen_prompt_match = re.search(r'\[GEN_PROMPT\](.*?)\[/GEN_PROMPT\]', contexts, re.DOTALL)
            question_match = re.search(r'\[QUESTION\](.*?)\[/QUESTION\]', contexts, re.DOTALL)

            if gen_prompt_match and question_match:
                # Use custom generation prompt from task config (WITHOUT question)
                generation_prompt = gen_prompt_match.group(1).strip()
                actual_question = question_match.group(1).strip()
                
                # Update contexts to be just the question for stage 2 (remove tags)
                contexts = contexts.replace(f"[GEN_PROMPT]{gen_prompt_match.group(1)}[/GEN_PROMPT]", "")
                contexts = contexts.replace(f"[QUESTION]{question_match.group(1)}[/QUESTION]", question_match.group(1))
                contexts = contexts.strip()  # Clean up whitespace
                eval_logger.info("Using custom generation prompt from task config (without question)")
            else:
                # Use default template
                generation_prompt = self.generation_prompt_template.format(question=contexts)

            eval_logger.info(f"\n{'='*60}")
            eval_logger.info(f"Processing doc {doc_id} from task {task}")
            eval_logger.info(f"{'='*60}")

            # Stage 1: Generate visualization image (REQUIRES original image as input)
            stage1_text, generated_images = self._stage1_generate_image(
                generation_prompt=generation_prompt,
                doc_id=doc_id,
                task=task,
                original_image=original_image,  # MUST pass original image
            )

            # Stage 2: Answer question 
            # If no auxiliary image was generated, still use original image to answer
            # Note: contexts has been cleaned to contain only the question text (GEN_PROMPT tags removed)
            if not generated_images or len(generated_images) == 0:
                eval_logger.warning(
                    f"No auxiliary image generated for doc {doc_id}, using original image only in Stage 2"
                )
                # Stage 2 with original image only (no auxiliary)
                final_answer = self._stage2_answer_with_image(
                    question=contexts,
                    image_path=None,  # No auxiliary image
                    doc_id=doc_id,
                    original_image=original_image  # Only original image
                )
            else:
                # Stage 2 with both original and auxiliary images
                # Image order: [original_image (primary), auxiliary_image (reference)]
                final_answer = self._stage2_answer_with_image(
                    question=contexts,  # Clean question text without tags
                    image_path=generated_images[0],  # Generated auxiliary/visualization image
                    doc_id=doc_id,
                    original_image=original_image  # Original image as primary reference
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

            # Return only final answer text (as per user requirement)
            res.append(final_answer)
            pbar.update(1)

        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """Not supported for generation models"""
        raise NotImplementedError(
            "MIOVisualCoT is a generation model and does not support loglikelihood"
        )

    def generate_until_multi_round(self, requests) -> List[str]:
        """Multi-round dialogue not yet implemented"""
        raise NotImplementedError(
            "Multi-round dialogue not yet implemented for MIOVisualCoT"
        )
