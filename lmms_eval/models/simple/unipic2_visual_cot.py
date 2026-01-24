import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

"""
UniPic2 Visual Chain-of-Thought (CoT) Model

This implementation uses UniPic2's full capabilities for visual reasoning:
1. Stage 1: Generate a visual diagram to help answer the question (using SD3.5 generation)
2. Stage 2: Answer the question using both the original image and generated diagram (using Qwen2.5-VL)

Prerequisites:
    1. Clone UniPic repository: git clone https://github.com/SkyworkAI/UniPic.git
    2. Install requirements: pip install -r UniPic-2/requirements.txt
    3. Ensure UniPic-2 is in your Python path

Example usage (loading from HuggingFace):
    python -m lmms_eval --model unipic2_visual_cot \
        --model_args pretrained=Skywork/UniPic2-Metaquery-9B,lmm_model=Qwen/Qwen2.5-VL-7B-Instruct \
        --tasks mme --batch_size 1 --device cuda:0
        
Example usage (loading from local path):
    python -m lmms_eval --model unipic2_visual_cot \
        --model_args pretrained=../models/UniPic2-Metaquery-9B,lmm_model=Qwen/Qwen2.5-VL-7B-Instruct \
        --tasks mme --batch_size 1 --device cuda:0
"""

# Add UniPic repository to Python path
wd = Path(__file__).parent.parent.parent.parent.resolve()
unipic_path = os.path.join(str(wd), "UniPic-2")
if os.path.exists(unipic_path):
    sys.path.insert(0, unipic_path)
    eval_logger.info(f"Added UniPic-2 path to sys.path: {unipic_path}")
else:
    # Try alternative path
    unipic_path_alt = os.path.join(str(wd), "UniPic", "UniPic-2")
    if os.path.exists(unipic_path_alt):
        sys.path.insert(0, unipic_path_alt)
        eval_logger.info(f"Added UniPic-2 path to sys.path: {unipic_path_alt}")
    else:
        eval_logger.warning(
            f"UniPic-2 repository not found at {unipic_path}. "
            f"Please clone it: cd {wd} && git clone https://github.com/SkyworkAI/UniPic.git"
        )


def fix_longer_edge(image: Image.Image, image_size: int = 512) -> Image.Image:
    """Resize image so the longer edge equals image_size while maintaining aspect ratio.
    Ensures dimensions are divisible by 16 for SD3.5 compatibility."""
    width, height = image.size
    if width > height:
        new_width = image_size
        new_height = int(height * image_size / width)
    else:
        new_height = image_size
        new_width = int(width * image_size / height)
    
    # Round to nearest multiple of 16 for SD3.5 compatibility
    new_width = (new_width // 16) * 16
    new_height = (new_height // 16) * 16
    
    # Ensure minimum size
    new_width = max(new_width, 16)
    new_height = max(new_height, 16)
    
    return image.resize((new_width, new_height), Image.LANCZOS)
    return image.resize((new_width, new_height), Image.LANCZOS)

@register_model("unipic2_visual_cot")
class UniPic2VisualCoT(lmms):
    def __init__(
        self,
        pretrained: str = "Skywork/UniPic2-Metaquery-9B",
        lmm_model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str = "cuda",
        dtype: str = "bfloat16",
        batch_size: int = 1,
        trust_remote_code: bool = True,
        attn_implementation: str = "flash_attention_2",
        image_size: int = 512,
        stage1_max_tokens: int = 2048,
        stage1_temperature: float = 0.7,
        stage1_num_inference_steps: int = 50,
        stage1_guidance_scale: float = 3.5,
        stage2_max_new_tokens: int = 1024,
        generation_prompt_template: str = "Generate a detailed visual diagram to help answer: {question}",
        output_dir: str = None,
        save_intermediate: bool = False,
        intermediate_dir: str = None,
        fail_gracefully: bool = True,
        system_prompt: str = "You are a helpful assistant.",
        remove_system_prompt: bool = False,
        infer_auto_device_map: bool = False,  # 新增：自动多 GPU 分配
        **kwargs,
    ) -> None:
        super().__init__()
        
        self.pretrained = pretrained
        self.lmm_model = lmm_model
        self.image_size = image_size
        self._device = device
        self._dtype = torch.bfloat16 if "bf16" in dtype else torch.float16
        self.save_intermediate = save_intermediate
        self.fail_gracefully = fail_gracefully
        self.generation_prompt_template = generation_prompt_template
        self.system_prompt = system_prompt
        self.remove_system_prompt = remove_system_prompt
        self.infer_auto_device_map = infer_auto_device_map
        
        if output_dir is None:
            self.output_dir = "./logs/unipic2_visual_cot"
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

        self.stage1_max_tokens = stage1_max_tokens
        self.stage1_temperature = stage1_temperature
        self.stage1_num_inference_steps = stage1_num_inference_steps
        self.stage1_guidance_scale = stage1_guidance_scale
        self.stage2_max_new_tokens = stage2_max_new_tokens

        from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
        
        # Load Qwen2.5-VL for understanding (directly from HuggingFace)
        eval_logger.info(f"Loading Qwen2.5-VL from HuggingFace: {lmm_model}...")
        
        # 决定 device_map
        if self.infer_auto_device_map:
            eval_logger.info("Using infer_auto_device_map for multi-GPU model parallelism")
            device_map_arg = "auto"
        else:
            device_map_arg = self._device
        
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            lmm_model,
            torch_dtype=self._dtype,
            attn_implementation=attn_implementation,
            trust_remote_code=trust_remote_code,
            device_map=device_map_arg,
        )
        self._processor = Qwen2_5_VLProcessor.from_pretrained(lmm_model, trust_remote_code=trust_remote_code)
        
        # Modify chat template (optional)
        if self.remove_system_prompt and self._processor.chat_template:
            self._processor.chat_template = self._processor.chat_template.replace(
                "{% if loop.first and message['role'] != 'system' %}"
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% endif %}",
                "",
            )
        
        # Load UniPic2 generation components (directly from HuggingFace)
        eval_logger.info(f"Loading UniPic2 generation pipeline from HuggingFace: {pretrained}...")
        self._load_generation_pipeline(pretrained)
        
        self.batch_size_per_gpu = int(batch_size)

    def _load_generation_pipeline(self, pretrained: str):
        """Load UniPic2 generation components for image generation from HuggingFace."""
        try:
            from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL
            
            # Check if UniPic-2 is available in sys.path
            try:
                from unipicv2.pipeline_stable_diffusion_3_kontext import StableDiffusion3KontextPipeline
                from unipicv2.transformer_sd3_kontext import SD3Transformer2DKontextModel
                from unipicv2.stable_diffusion_3_conditioner import StableDiffusion3Conditioner
            except ImportError:
                eval_logger.error(
                    "UniPic-2 modules not found. Please ensure UniPic-2 is in your Python path.\n"
                    "Clone it: git clone https://github.com/SkyworkAI/UniPic.git\n"
                    "And add UniPic-2 to sys.path or install it."
                )
                raise
            
            # Load from HuggingFace Hub directly
            eval_logger.info(f"Loading transformer from HuggingFace: {pretrained}/transformer...")
            self.transformer = SD3Transformer2DKontextModel.from_pretrained(
                pretrained, 
                subfolder="transformer", 
                torch_dtype=self._dtype,
                trust_remote_code=True
            )
            
            eval_logger.info(f"Loading vae from HuggingFace: {pretrained}/vae...")
            self.vae = AutoencoderKL.from_pretrained(
                pretrained, 
                subfolder="vae", 
                torch_dtype=self._dtype,
                trust_remote_code=True
            )
            
            eval_logger.info(f"Loading conditioner from HuggingFace: {pretrained}/conditioner...")
            self.conditioner = StableDiffusion3Conditioner.from_pretrained(
                pretrained, 
                subfolder="conditioner", 
                torch_dtype=self._dtype,
                trust_remote_code=True
            )
            
            # Device allocation strategy
            if self.infer_auto_device_map and torch.cuda.device_count() > 1:
                # Multi-GPU: distribute components across available GPUs
                num_gpus = torch.cuda.device_count()
                eval_logger.info(f"Multi-GPU mode: distributing UniPic2 components across {num_gpus} visible GPUs")
                
                # Get the first available GPU (respects CUDA_VISIBLE_DEVICES)
                first_gpu = torch.cuda.current_device()
                
                # Put all components on the first GPU initially
                # They will be moved dynamically to match Qwen's output device
                self.transformer = self.transformer.to(f"cuda:{first_gpu}")
                self.vae = self.vae.to(f"cuda:{first_gpu}")
                self.conditioner = self.conditioner.to(f"cuda:{first_gpu}")
                
                eval_logger.info(f"  - All UniPic2 components on cuda:{first_gpu}")
                eval_logger.info(f"  - Components will be moved dynamically to match Qwen output device")
            else:
                # Single GPU: put everything on the same device
                self.transformer = self.transformer.to(self._device)
                self.vae = self.vae.to(self._device)
                self.conditioner = self.conditioner.to(self._device)
                eval_logger.info(f"Single GPU mode: all components on {self._device}")
            
            eval_logger.info(f"Loading scheduler from HuggingFace: {pretrained}/scheduler...")
            self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                pretrained, 
                subfolder="scheduler",
                trust_remote_code=True
            )
            
            # Create pipeline
            self.pipeline = StableDiffusion3KontextPipeline(
                transformer=self.transformer,
                vae=self.vae,
                text_encoder=None,
                tokenizer=None,
                text_encoder_2=None,
                tokenizer_2=None,
                text_encoder_3=None,
                tokenizer_3=None,
                scheduler=self.scheduler,
            )
            
            eval_logger.info("UniPic2 generation pipeline loaded successfully from HuggingFace")
            
        except Exception as e:
            eval_logger.error(f"Failed to load generation pipeline: {e}")
            if self.fail_gracefully:
                eval_logger.warning("Continuing without generation capability")
                self.pipeline = None
            else:
                raise

    def _extract_image_from_various_formats(self, img_data) -> Optional[Image.Image]:
        """
        Extract PIL Image from various formats (HuggingFace datasets, file paths, etc.)
        
        Args:
            img_data: Image data in various formats
            
        Returns:
            PIL Image or None if extraction fails
        """
        try:
            if img_data is None:
                return None
            elif isinstance(img_data, Image.Image):
                return img_data.convert("RGB")
            elif isinstance(img_data, str):
                # File path
                return Image.open(img_data).convert("RGB")
            elif isinstance(img_data, dict):
                # HuggingFace dataset format
                if "bytes" in img_data:
                    from io import BytesIO
                    return Image.open(BytesIO(img_data["bytes"])).convert("RGB")
                elif "path" in img_data:
                    return Image.open(img_data["path"]).convert("RGB")
                elif "image" in img_data:
                    inner_img = img_data["image"]
                    return self._extract_image_from_various_formats(inner_img)
            else:
                # Try to open it directly
                return Image.open(img_data).convert("RGB")
        except Exception as e:
            eval_logger.debug(f"Failed to extract image from format {type(img_data)}: {e}")
            return None

    def _stage1_generate_image(self, generation_prompt: str, doc_id: str, task: str, original_image=None) -> Tuple[str, List[str]]:
        """Stage 1: Generate visual diagram using UniPic2 generation pipeline."""
        if self.pipeline is None:
            eval_logger.warning("Generation pipeline not available, skipping image generation")
            return "", []
            
        try:
            # Convert original_image to PIL Image using the robust extraction method
            if original_image is not None:
                original_image = self._extract_image_from_various_formats(original_image)
                if original_image is None:
                    eval_logger.warning(f"Failed to extract original image for doc {doc_id}")
            
            # Prepare prompts
            prompt = generation_prompt
            negative_prompt = "blurry, low quality, low resolution, distorted, deformed, broken content, missing parts, damaged details, artifacts, glitch, noise, pixelated, grainy, compression artifacts, bad composition, wrong proportion, incomplete editing, unfinished, unedited areas."
            
            # Build messages for prompt encoding
            messages = []
            if not self.remove_system_prompt:
                messages.append([{"role": "system", "content": self.system_prompt}])
            else:
                messages.append([])
            
            # Add prompt and negative prompt
            for txt in [prompt, negative_prompt]:
                msg_content = []
                if original_image:
                    msg_content.append({"type": "image", "image": fix_longer_edge(original_image, self.image_size)})
                msg_content.append({"type": "text", "text": f"Generate an image: {txt}"})
                
                msg = messages[0].copy() if messages[0] else []
                msg.append({"role": "user", "content": msg_content})
                messages.append(msg)
            
            # Process with Qwen2.5-VL to get embeddings
            texts = [self._processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages[1:]]
            
            # Determine target device for inputs
            if self.infer_auto_device_map:
                # For multi-GPU, use the device of the first model parameter
                target_device = next(self._model.parameters()).device
            else:
                target_device = self._device
            
            if original_image:
                min_pixels = max_pixels = int(original_image.height * 28 / 32 * original_image.width * 28 / 32)
                inputs = self._processor(
                    text=texts,
                    images=[fix_longer_edge(original_image, self.image_size)] * 2,
                    min_pixels=min_pixels,
                    max_pixels=max_pixels,
                    videos=None,
                    padding=True,
                    return_tensors="pt"
                ).to(target_device)
            else:
                inputs = self._processor(
                    text=texts,
                    images=None,
                    videos=None,
                    padding=True,
                    return_tensors="pt"
                ).to(target_device)
            
            # Wrap in no_grad to save memory
            with torch.no_grad():
                # Get embeddings from LMM
                input_ids = inputs.input_ids
                attention_mask = inputs.attention_mask
                
                # Add meta queries
                input_ids = torch.cat([input_ids, input_ids.new_zeros(2, self.conditioner.config.num_queries)], dim=1)
                attention_mask = torch.cat([attention_mask, attention_mask.new_ones(2, self.conditioner.config.num_queries)], dim=1)
                inputs_embeds = self._model.get_input_embeddings()(input_ids)
                inputs_embeds[:, -self.conditioner.config.num_queries:] = self.conditioner.meta_queries[None].expand(2, -1, -1)
                
                # Process with vision if image exists
                if original_image and "pixel_values" in inputs:
                    image_embeds = self._model.visual(inputs.pixel_values, grid_thw=inputs.image_grid_thw)
                    image_token_id = self._processor.tokenizer.convert_tokens_to_ids('<|image_pad|>')
                    inputs_embeds[input_ids == image_token_id] = image_embeds
                    
                    self._model.model.rope_deltas = None
                    outputs = self._model.model(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        image_grid_thw=inputs.image_grid_thw,
                        use_cache=False
                    )
                else:
                    outputs = self._model.model(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        use_cache=False
                    )
            
            hidden_states = outputs.last_hidden_state[:, -self.conditioner.config.num_queries:]
            
            # Move conditioner to match hidden_states device (important for multi-GPU)
            hidden_states_device = hidden_states.device
            conditioner_device = next(self.conditioner.parameters()).device
            
            if hidden_states_device != conditioner_device:
                eval_logger.info(f"Moving conditioner from {conditioner_device} to {hidden_states_device}")
                self.conditioner = self.conditioner.to(hidden_states_device)
            
            prompt_embeds, pooled_prompt_embeds = self.conditioner(hidden_states)
            
            # Move embeddings to transformer device for pipeline
            transformer_device = next(self.transformer.parameters()).device
            prompt_embeds = prompt_embeds.to(transformer_device)
            pooled_prompt_embeds = pooled_prompt_embeds.to(transformer_device)
            
            # Generate image
            if original_image:
                # Image editing mode - resize image first
                resized_image = fix_longer_edge(original_image, self.image_size)
                # Use resized image dimensions (already divisible by 16)
                height = resized_image.height
                width = resized_image.width
                
                generated_image = self.pipeline(
                    image=resized_image,
                    prompt_embeds=prompt_embeds[:1],
                    pooled_prompt_embeds=pooled_prompt_embeds[:1],
                    negative_prompt_embeds=prompt_embeds[1:],
                    negative_pooled_prompt_embeds=pooled_prompt_embeds[1:],
                    height=height,
                    width=width,
                    num_inference_steps=self.stage1_num_inference_steps,
                    guidance_scale=self.stage1_guidance_scale,
                    generator=torch.Generator(device=transformer_device).manual_seed(42)
                ).images[0]
            else:
                # Text-to-image mode
                height = 512
                width = 512
                
                generated_image = self.pipeline(
                    prompt_embeds=prompt_embeds[:1],
                    pooled_prompt_embeds=pooled_prompt_embeds[:1],
                    negative_prompt_embeds=prompt_embeds[1:],
                    negative_pooled_prompt_embeds=pooled_prompt_embeds[1:],
                    height=height,
                    width=width,
                    num_inference_steps=self.stage1_num_inference_steps,
                    guidance_scale=self.stage1_guidance_scale,
                    generator=torch.Generator(device=transformer_device).manual_seed(42)
                ).images[0]

            # Save generated image
            task_dir = os.path.join(self.generated_images_dir, task)
            os.makedirs(task_dir, exist_ok=True)
            image_path = os.path.join(task_dir, f"{doc_id}_gen.png")
            generated_image.save(image_path)
            
            # Clean up intermediate tensors to free memory
            del inputs, input_ids, attention_mask, inputs_embeds, outputs, hidden_states
            del prompt_embeds, pooled_prompt_embeds
            if 'image_embeds' in locals():
                del image_embeds
            torch.cuda.empty_cache()
            
            eval_logger.info(f"Generated image saved to {image_path}")
            return "Success", [image_path]

        except Exception as e:
            eval_logger.error(f"Stage 1 generation error: {e}")
            import traceback
            eval_logger.error(traceback.format_exc())
            if self.fail_gracefully:
                return "", []
            raise e

    def _stage2_answer_with_image(self, question: str, image_path: str, original_image=None) -> str:
        """Stage 2: Answer question using generated image + original image."""
        try:
            # Load generated image
            gen_img = Image.open(image_path).convert("RGB")
            
            # Convert original_image using the robust extraction method
            if original_image is not None:
                original_image = self._extract_image_from_various_formats(original_image)
            
            # Build images list
            images = []
            if original_image:
                images.append(original_image)
            images.append(gen_img)
            
            # Build content with properly converted images
            content = []
            for img in images:
                content.append({"type": "image", "image": fix_longer_edge(img, self.image_size)})
            content.append({"type": "text", "text": question})
            
            messages = []
            if not self.remove_system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.append({"role": "user", "content": content})
            
            text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            # Determine target device for inputs
            if self.infer_auto_device_map:
                target_device = next(self._model.parameters()).device
            else:
                target_device = self._device
            
            inputs = self._processor(text=[text], images=images, return_tensors="pt").to(target_device)

            with torch.no_grad():
                output = self._model.generate(**inputs, max_new_tokens=self.stage2_max_new_tokens, do_sample=False)
            
            input_len = inputs["input_ids"].shape[1]
            generated_ids = output[0][input_len:]
            response = self._processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # Clean up
            del output, inputs, generated_ids
            torch.cuda.empty_cache()
            
            return response
            
        except Exception as e:
            eval_logger.error(f"Stage 2 error: {e}")
            import traceback
            eval_logger.error(traceback.format_exc())
            if self.fail_gracefully:
                return ""
            raise e

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
        Uni-MMMU interleaved generation for UniPic2 Visual CoT.

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

            # Use stage 2 to answer with the generated images
            if len(generated_images) >= 2:
                # Load both generated images
                gen_img0 = Image.open(generated_images[0]).convert("RGB")
                gen_img1 = Image.open(generated_images[1]).convert("RGB")

                # Build images list
                images = []
                if original_image:
                    images.append(original_image)
                images.extend([gen_img0, gen_img1])

                # Build content with properly converted images
                content = []
                for img in images:
                    content.append({"type": "image", "image": fix_longer_edge(img, self.image_size)})
                content.append({"type": "text", "text": final_question})

                messages = []
                if not self.remove_system_prompt:
                    messages.append({"role": "system", "content": self.system_prompt})
                messages.append({"role": "user", "content": content})

                text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

                # Determine target device for inputs
                if self.infer_auto_device_map:
                    target_device = next(self._model.parameters()).device
                else:
                    target_device = self._device

                inputs = self._processor(text=[text], images=images, return_tensors="pt").to(target_device)

                with torch.no_grad():
                    output = self._model.generate(**inputs, max_new_tokens=self.stage2_max_new_tokens, do_sample=False)

                input_len = inputs["input_ids"].shape[1]
                generated_ids = output[0][input_len:]
                final_text = self._processor.tokenizer.decode(generated_ids, skip_special_tokens=True)

                # Clean up
                del output, inputs, generated_ids
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

            # Use stage 2 to answer with all generated images
            if generated_images:
                # Load all generated images
                step_images = [Image.open(img_path).convert("RGB") for img_path in generated_images]

                # Build images list
                images = []
                if original_image:
                    images.append(original_image)
                images.extend(step_images)

                # Build content with properly converted images
                content = []
                for img in images:
                    content.append({"type": "image", "image": fix_longer_edge(img, self.image_size)})
                content.append({"type": "text", "text": final_question})

                messages = []
                if not self.remove_system_prompt:
                    messages.append({"role": "system", "content": self.system_prompt})
                messages.append({"role": "user", "content": content})

                text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

                # Determine target device for inputs
                if self.infer_auto_device_map:
                    target_device = next(self._model.parameters()).device
                else:
                    target_device = self._device

                inputs = self._processor(text=[text], images=images, return_tensors="pt").to(target_device)

                with torch.no_grad():
                    output = self._model.generate(**inputs, max_new_tokens=self.stage2_max_new_tokens, do_sample=False)

                input_len = inputs["input_ids"].shape[1]
                generated_ids = output[0][input_len:]
                final_text = self._processor.tokenizer.decode(generated_ids, skip_special_tokens=True)

                # Clean up
                del output, inputs, generated_ids
                torch.cuda.empty_cache()
            else:
                final_text = ""

        return final_text, generated_images

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []
        for request in tqdm(requests, desc="UniPic2 CoT Reasoning"):
            contexts, gen_kwargs, doc_to_visual, doc_id, task, split = request.args

            # Check if this is Uni-MMMU interleaved generation mode
            bagel_interleaved = gen_kwargs.get("bagel_interleaved", None)

            if bagel_interleaved is not None:
                # Uni-MMMU interleaved generation mode
                eval_logger.info(f"Uni-MMMU interleaved mode for doc {doc_id}")

                # Get input images and doc data
                doc = self.task_dict[task][split][doc_id]
                input_images = []
                if doc_to_visual:
                    visuals = doc_to_visual(doc)
                    if visuals:
                        input_images = visuals if isinstance(visuals, list) else [visuals]

                # Generate using interleaved mode
                final_ans, gen_paths = self.generate_uni_mmmu_interleaved(
                    input_images, contexts, str(doc_id), task, bagel_interleaved, doc
                )

                # Save intermediate artifacts if enabled
                self._save_intermediate_artifacts(
                    doc_id=str(doc_id),
                    task=task,
                    generation_prompt=f"Interleaved generation: {bagel_interleaved.get('task_type', 'unknown')}",
                    stage1_text="",
                    generated_images=gen_paths,
                    question=contexts,
                    stage2_answer=final_ans,
                )

                res.append(final_ans)
            else:
                # Standard single-image generation mode
                original_image = None
                if doc_to_visual:
                    doc = self.task_dict[task][split][doc_id]
                    visuals = doc_to_visual(doc)
                    if visuals: original_image = visuals[0]

                gen_prompt = self.generation_prompt_template.format(question=contexts)
                stage1_text, gen_paths = self._stage1_generate_image(gen_prompt, doc_id, task, original_image)

                if gen_paths:
                    final_ans = self._stage2_answer_with_image(contexts, gen_paths[0], original_image)
                else:
                    final_ans = ""

                # Save intermediate artifacts if enabled
                self._save_intermediate_artifacts(
                    doc_id=doc_id,
                    task=task,
                    generation_prompt=gen_prompt,
                    stage1_text=stage1_text,
                    generated_images=gen_paths,
                    question=contexts,
                    stage2_answer=final_ans,
                )

                res.append(final_ans)

            torch.cuda.empty_cache()

        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("UniPic2 is a generation model.")

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("Multi-round not implemented.")


