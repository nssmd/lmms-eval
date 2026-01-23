"""
Emu3 Visual Chain-of-Thought Model

Two-stage inference:
1. Stage 1: Generate auxiliary image from original image + text prompt
2. Stage 2: Answer question using original image + auxiliary image + text

Emu3 treats both images and text as tokens, enabling image-conditioned generation.

Usage:
    python -m lmms_eval \
        --model emu3_visual_cot \
        --model_args pretrained=BAAI/Emu3-Chat \
        --tasks mathvista_visual_cot \
        --batch_size 1 \
        --output_path ./logs/
"""

import json
import os
from typing import List, Optional, Tuple

import torch
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

# Patch DynamicCache to support seen_tokens property for compatibility
# with older EMU3 model code that uses seen_tokens instead of get_seq_length()
from transformers.cache_utils import DynamicCache

if not hasattr(DynamicCache, "seen_tokens"):
    # Add seen_tokens as a property for backward compatibility
    DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())
    eval_logger.info("Patched DynamicCache with seen_tokens property for EMU3 compatibility")

if not hasattr(DynamicCache, "get_max_length"):
    # Add get_max_length method for backward compatibility
    # Returns None since DynamicCache doesn't have a fixed max length
    DynamicCache.get_max_length = lambda self: None
    eval_logger.info("Patched DynamicCache with get_max_length method for EMU3 compatibility")

if not hasattr(DynamicCache, "get_usable_length"):
    # Add get_usable_length method for backward compatibility
    # Returns the current sequence length (same as get_seq_length)
    DynamicCache.get_usable_length = lambda self, new_seq_length, layer_idx=0: self.get_seq_length(layer_idx)
    eval_logger.info("Patched DynamicCache with get_usable_length method for EMU3 compatibility")


@register_model("emu3_visual_cot")
class Emu3VisualCoT(lmms):
    """
    Emu3 Visual Chain-of-Thought Model

    Performs two-stage visual reasoning:
    1. Stage 1: Generate auxiliary image from original image + text prompt
    2. Stage 2: Answer question using original image + auxiliary image + text

    Emu3 uses VQ-VAE to tokenize images into discrete tokens, enabling unified
    next-token prediction for both understanding and generation.
    """

    def __init__(
        self,
        pretrained: str = "BAAI/Emu3-Chat",
        vq_hub: str = "BAAI/Emu3-VisionTokenizer",
        device: str = "cuda",
        device_map: str = "auto",
        batch_size: int = 1,
        # Stage 1: Image generation parameters
        stage1_max_new_tokens: int = 6000,  # ~4096 visual tokens + overhead
        stage1_image_height: int = 256,  # Smaller for faster generation
        stage1_image_width: int = 256,
        stage1_negative_prompt: str = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry.",
        # Stage 2: Text generation parameters
        stage2_max_new_tokens: int = 512,
        stage2_do_sample: bool = False,
        stage2_temperature: float = 0.0,
        stage2_top_p: float = 1.0,
        # Generation prompt template
        generation_prompt_template: str = "Based on this image, generate a detailed visual diagram or illustration to help answer this question: {question}",
        # Model loading
        use_flash_attention_2: bool = True,
        torch_dtype: str = "bfloat16",
        # Output and debugging
        output_dir: Optional[str] = None,
        save_intermediate: bool = True,
        intermediate_dir: Optional[str] = None,
        # Error handling
        fail_gracefully: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()

        self.pretrained = pretrained
        self.vq_hub = vq_hub
        self.device_str = device
        self.device_map = device_map
        self.batch_size_per_gpu = batch_size

        # Stage 1 parameters
        self.stage1_max_new_tokens = stage1_max_new_tokens
        self.stage1_image_height = stage1_image_height
        self.stage1_image_width = stage1_image_width
        self.stage1_negative_prompt = stage1_negative_prompt

        # Stage 2 parameters
        self.stage2_max_new_tokens = stage2_max_new_tokens
        self.stage2_do_sample = stage2_do_sample
        self.stage2_temperature = stage2_temperature
        self.stage2_top_p = stage2_top_p

        self.generation_prompt_template = generation_prompt_template
        self.use_flash_attention_2 = use_flash_attention_2
        self.fail_gracefully = fail_gracefully

        # Set torch dtype
        if torch_dtype == "bfloat16":
            self.torch_dtype = torch.bfloat16
        elif torch_dtype == "float16":
            self.torch_dtype = torch.float16
        else:
            self.torch_dtype = torch.float32

        # Setup output directories
        if output_dir is None:
            self.output_dir = "./logs/emu3_visual_cot"
        else:
            self.output_dir = output_dir

        if intermediate_dir is None:
            self.intermediate_dir = self.output_dir
        else:
            self.intermediate_dir = intermediate_dir

        self.save_intermediate = save_intermediate
        if save_intermediate:
            os.makedirs(self.intermediate_dir, exist_ok=True)
            eval_logger.info(
                f"Intermediate artifacts will be saved under: {self.intermediate_dir}"
            )

        # Load model and processor
        eval_logger.info(f"Loading Emu3 model from {pretrained}")
        self._load_model()

        # Compute visual tokens from tokenizer vocabulary
        self.visual_tokens = self._get_visual_tokens()
        eval_logger.info(f"Loaded {len(self.visual_tokens)} visual tokens")

        eval_logger.info("Emu3VisualCoT initialized successfully")

    def _load_model(self):
        """Load Emu3 model, vision tokenizer, and processor"""
        from transformers import (
            AutoImageProcessor,
            AutoModel,
            AutoModelForCausalLM,
            AutoTokenizer,
        )

        # Setup attention implementation
        attn_implementation = (
            "flash_attention_2" if self.use_flash_attention_2 else "eager"
        )
        if self.use_flash_attention_2:
            try:
                import flash_attn  # noqa: F401

                eval_logger.info("Using Flash Attention 2")
            except ImportError:
                eval_logger.warning(
                    "flash_attn not installed, falling back to eager attention"
                )
                attn_implementation = "eager"

        # Determine device
        if self.device_map == "auto":
            device = "cuda:0"
        else:
            device = self.device_str if torch.cuda.is_available() else "cpu"
        self.device = device

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.pretrained, trust_remote_code=True, padding_side="left"
        )

        # Load image processor and vision tokenizer
        eval_logger.info(f"Loading vision tokenizer from {self.vq_hub}")
        self.image_processor = AutoImageProcessor.from_pretrained(
            self.vq_hub, trust_remote_code=True
        )
        self.image_tokenizer = (
            AutoModel.from_pretrained(
                self.vq_hub, device_map=device, trust_remote_code=True
            )
            .eval()
        )

        # Import and create custom Emu3Processor from model's code
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        Emu3Processor = get_class_from_dynamic_module(
            "processing_emu3.Emu3Processor", self.pretrained
        )
        self.processor = Emu3Processor(
            self.image_processor, self.image_tokenizer, self.tokenizer
        )

        # Load main model
        eval_logger.info(f"Loading main model from {self.pretrained}")
        if self.device_map == "auto":
            self.model = (
                AutoModelForCausalLM.from_pretrained(
                    self.pretrained,
                    device_map="auto",
                    torch_dtype=self.torch_dtype,
                    attn_implementation=attn_implementation,
                    trust_remote_code=True,
                )
                .eval()
            )
            eval_logger.info("Model loaded with automatic device mapping")
        else:
            self.model = (
                AutoModelForCausalLM.from_pretrained(
                    self.pretrained,
                    torch_dtype=self.torch_dtype,
                    attn_implementation=attn_implementation,
                    trust_remote_code=True,
                )
                .to(device)
                .eval()
            )
            eval_logger.info(f"Model loaded on {device}")

    def _get_visual_tokens(self):
        """
        Get visual tokens from the processor's const_helper.

        Emu3 uses VQ-VAE to tokenize images into discrete tokens.
        The visual tokens are in a specific range of the vocabulary (151854-184621).
        """
        # Use const_helper to get the correct visual tokens range
        # This is the authoritative source for visual token IDs
        const_helper = self.processor.const_helper(height=32, width=32)
        visual_tokens = tuple(const_helper.visual_tokens)

        eval_logger.debug(
            f"Got {len(visual_tokens)} visual tokens from const_helper "
            f"(range: {visual_tokens[0]}-{visual_tokens[-1]})"
        )

        return visual_tokens

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    def flatten(self, input_list):
        """Flatten nested lists"""
        if not input_list or any(i is None for i in input_list):
            return []
        output = []
        for item in input_list:
            if isinstance(item, list):
                output.extend(self.flatten(item))
            else:
                output.append(item)
        return output

    def _resize_image(self, image: Image.Image, max_size: int = 512) -> Image.Image:
        """Resize image to max_size x max_size to save memory"""
        if image.width > max_size or image.height > max_size:
            image = image.resize((max_size, max_size), Image.Resampling.LANCZOS)
        return image

    def _decode_image_tokens(
        self, token_ids: torch.Tensor, height: int, width: int
    ) -> List[Image.Image]:
        """
        Decode image tokens to PIL Images

        Args:
            token_ids: Generated token IDs [B, seq_len]
            height: Expected image height in tokens
            width: Expected image width in tokens

        Returns:
            List of PIL Images
        """
        # Get special token IDs
        eol_id = self.tokenizer.encode("<|extra_200|>", add_special_tokens=False)[0]
        eof_id = getattr(self.tokenizer, "eof_token_id", None)
        eoi_id = getattr(self.tokenizer, "eoi_token_id", None)
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id

        # Get visual token range and compute offset for codebook conversion
        visual_tokens = self.visual_tokens
        visual_tokens_set = set(visual_tokens)
        # Offset to convert token IDs to codebook indices
        # Token IDs are 151854-184621, codebook indices are 0-32767
        visual_token_offset = min(visual_tokens)

        batch_images = []
        for batch_idx in range(token_ids.shape[0]):
            seq = token_ids[batch_idx]

            # Extract visual tokens (filter out special tokens)
            visual_token_list = []
            for token_id in seq:
                token_val = token_id.item()
                # Stop at end tokens
                if token_val in [eof_id, eoi_id, eos_id, pad_id]:
                    break
                # Skip EOL tokens
                if token_val == eol_id:
                    continue
                # Collect visual tokens (convert to codebook indices)
                if token_val in visual_tokens_set:
                    # Convert token ID to codebook index
                    codebook_idx = token_val - visual_token_offset
                    visual_token_list.append(codebook_idx)

            # Reshape to (height, width)
            expected_tokens = height * width
            if len(visual_token_list) < expected_tokens:
                eval_logger.warning(
                    f"Expected {expected_tokens} visual tokens, got {len(visual_token_list)}"
                )
                # Pad with first codebook index (0) if needed
                visual_token_list.extend(
                    [0] * (expected_tokens - len(visual_token_list))
                )
            elif len(visual_token_list) > expected_tokens:
                visual_token_list = visual_token_list[:expected_tokens]

            # Convert to tensor and reshape
            visual_tensor = torch.tensor(
                visual_token_list, dtype=torch.long, device=self.image_tokenizer.device
            ).reshape(height, width)

            # Decode using vision tokenizer
            decoded_image = self.image_tokenizer.decode(visual_tensor.unsqueeze(0))

            # Post-process to PIL Image
            decoded_image = decoded_image.float()
            processed = self.image_processor.postprocess(decoded_image)
            pil_image = processed["pixel_values"][0]

            batch_images.append(pil_image)

        return batch_images

    def _stage1_generate_image(
        self,
        generation_prompt: str,
        original_image: Image.Image,
        doc_id: str,
        task: str,
    ) -> Tuple[str, List[str]]:
        """
        Stage 1: Generate auxiliary image from original image + text prompt

        This method implements image-conditioned generation by:
        1. Tokenizing the original image using mode='U'
        2. Appending image generation tokens
        3. Generating new image tokens

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
            # Ensure image is RGB and resize
            if hasattr(original_image, "mode") and original_image.mode != "RGB":
                original_image = original_image.convert("RGB")
            original_image = self._resize_image(original_image, max_size=512)

            # Step 1: Process original image + text using mode='U' to get image tokens
            # This gives us the input with the original image tokenized
            inputs_understanding = self.processor(
                text=generation_prompt,
                image=original_image,
                mode="U",
                return_tensors="pt",
                padding="longest",
            )

            device = self.model.device
            input_ids = inputs_understanding.input_ids.to(device)
            attention_mask = inputs_understanding.attention_mask.to(device)

            # Step 2: Append image generation tokens
            # Get the image generation prefix tokens
            HEIGHT, WIDTH = self.stage1_image_height, self.stage1_image_width
            # Calculate token dimensions (Emu3 uses 8x downsampling)
            spatial_scale = getattr(
                self.processor.vision_tokenizer, "spatial_scale_factor", 8
            )
            h_tokens = HEIGHT // spatial_scale
            w_tokens = WIDTH // spatial_scale

            eval_logger.info(
                f"Image generation: HEIGHT={HEIGHT}, WIDTH={WIDTH}, "
                f"spatial_scale={spatial_scale}, h_tokens={h_tokens}, w_tokens={w_tokens}, "
                f"expected_tokens={h_tokens * (w_tokens + 1) + 3}"
            )

            # Build image generation prompt tokens
            image_gen_prompt = (
                self.tokenizer.boi_token
                + self.processor.prefix_template.format(H=h_tokens, W=w_tokens)
                + self.tokenizer.img_token
            )
            gen_tokens = self.tokenizer.encode(
                image_gen_prompt, add_special_tokens=False, return_tensors="pt"
            ).to(device)

            # Concatenate understanding input with generation tokens
            input_ids = torch.cat([input_ids, gen_tokens], dim=1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones_like(gen_tokens)], dim=1
            )
            input_length = input_ids.shape[1]

            # Step 3: Prepare negative prompt
            neg_inputs = self.processor(
                text=self.stage1_negative_prompt,
                mode="G",
                ratio=f"{WIDTH}:{HEIGHT}",
                return_tensors="pt",
            )
            neg_input_ids = neg_inputs.input_ids.to(device)
            neg_attention_mask = neg_inputs.attention_mask.to(device)

            # Step 4: Build custom prefix constraint function
            # Note: We can't use processor.build_prefix_constrained_fn directly because
            # it finds the FIRST img_token, but we have TWO (original image + generated image)
            # We need to use the LAST img_token position

            # Get token IDs from processor's const_helper
            const_helper = self.processor.const_helper(height=h_tokens, width=w_tokens)
            img_token = const_helper.img_token
            eol_token = const_helper.eol_token
            eof_token = const_helper.eof_token
            eoi_token = const_helper.eoi_token
            eos_token = const_helper.eos_token
            pad_token = const_helper.pad_token
            visual_tokens = const_helper.visual_tokens

            eval_logger.info(
                f"Token IDs - img: {img_token}, eol: {eol_token}, eof: {eof_token}, "
                f"eoi: {eoi_token}, eos: {eos_token}, visual_tokens range: {visual_tokens[0]}-{visual_tokens[-1]}"
            )

            # Cache for the position of the LAST img_token
            offset_cache = {}
            call_count = [0]

            def prefix_allowed_tokens_fn(batch_id, input_ids):
                call_count[0] += 1

                if batch_id not in offset_cache:
                    # Find the LAST img_token position (for the generated image)
                    positions = torch.nonzero(input_ids == img_token, as_tuple=True)[0]
                    if len(positions) > 0:
                        offset_cache[batch_id] = positions[-1].item()  # Use LAST position
                    else:
                        offset_cache[batch_id] = 0
                    eval_logger.info(f"First call: img_token position = {offset_cache[batch_id]}, input_ids len = {input_ids.shape[0]}")

                offset = input_ids.shape[0] - offset_cache[batch_id]

                # Log first few calls and periodically
                if call_count[0] <= 5 or call_count[0] % 200 == 0:
                    result_type = "eol" if offset % (w_tokens + 1) == 0 else \
                                  "eof" if offset == (w_tokens + 1) * h_tokens + 1 else \
                                  "eoi" if offset == (w_tokens + 1) * h_tokens + 2 else \
                                  "eos" if offset == (w_tokens + 1) * h_tokens + 3 else \
                                  "pad" if offset > (w_tokens + 1) * h_tokens + 3 else "visual"
                    eval_logger.info(f"Call {call_count[0]}: offset={offset}, returning {result_type}")

                if offset % (w_tokens + 1) == 0:
                    return (eol_token,)
                elif offset == (w_tokens + 1) * h_tokens + 1:
                    return (eof_token,)
                elif offset == (w_tokens + 1) * h_tokens + 2:
                    return (eoi_token,)
                elif offset == (w_tokens + 1) * h_tokens + 3:
                    return (eos_token,)
                elif offset > (w_tokens + 1) * h_tokens + 3:
                    return (pad_token,)
                else:
                    return visual_tokens

            # Step 5: Generate image tokens
            eval_logger.info(f"Generating {h_tokens}x{w_tokens} image tokens...")

            # Note: Negative prompt may not work well for image-conditioned generation
            # Try without it first
            with torch.no_grad():
                out = self.model.generate(
                    input_ids,
                    max_new_tokens=self.stage1_max_new_tokens,
                    attention_mask=attention_mask,
                    prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                    # negative_prompt_ids=neg_input_ids,
                    # negative_prompt_attention_mask=neg_attention_mask,
                    return_dict_in_generate=True,
                    use_cache=True,  # Enable KV cache for faster generation
                )

            eval_logger.info(f"Generated {out.sequences.shape[1] - input_length} tokens")

            # Step 6: Decode image tokens
            # Extract visual tokens from generated sequence
            generated_tokens = out.sequences[:, input_length:]

            # Debug: Check what tokens were actually generated
            unique_tokens = torch.unique(generated_tokens).tolist()
            eval_logger.info(f"Unique generated token IDs (first 20): {unique_tokens[:20]}")
            eval_logger.info(f"Total unique tokens: {len(unique_tokens)}")

            # Check how many are visual tokens
            visual_tokens_set = set(self.visual_tokens)
            visual_count = sum(1 for t in generated_tokens[0] if t.item() in visual_tokens_set)
            eval_logger.info(f"Visual tokens in generated sequence: {visual_count}/{generated_tokens.shape[1]}")

            images = self._decode_image_tokens(generated_tokens, h_tokens, w_tokens)

            # Step 7: Save generated images
            output_images = []
            task_dir = os.path.join(self.intermediate_dir, task)
            os.makedirs(task_dir, exist_ok=True)

            for i, img in enumerate(images):
                safe_filename = f"{doc_id}_stage1_{i}.png"
                image_path = os.path.join(task_dir, safe_filename)
                img.save(image_path)
                output_images.append(image_path)
                eval_logger.info(f"Saved generated auxiliary image: {image_path}")

            # Clean up
            del inputs_understanding, input_ids, attention_mask, out
            torch.cuda.empty_cache()

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
        Stage 2: Answer question using original image + auxiliary image + text

        This method processes multiple images by concatenating their tokens.

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
            # Load auxiliary image
            auxiliary_image = Image.open(auxiliary_image_path).convert("RGB")

            # Resize both images
            original_image = self._resize_image(original_image, max_size=512)
            auxiliary_image = self._resize_image(auxiliary_image, max_size=512)

            # Build prompt with both images
            # Format: [Original Image] [Auxiliary Image] Question
            prompt_with_images = (
                "Here is the original image and an auxiliary visualization. "
                "Based on both images, please answer the following question:\n"
                f"{question}"
            )

            # Tokenize both images
            original_tokens = self.processor.tokenize_image(
                [original_image], padding_image=False
            )[0]
            auxiliary_tokens = self.processor.tokenize_image(
                [auxiliary_image], padding_image=False
            )[0]

            # Build input with both images
            h1, w1 = original_tokens.shape
            h2, w2 = auxiliary_tokens.shape

            imgstr1 = self.processor.to_imgstr(original_tokens)
            imgstr2 = self.processor.to_imgstr(auxiliary_tokens)

            # Build image prompts
            image_prompt1 = (
                self.tokenizer.boi_token
                + self.processor.prefix_template.format(H=h1, W=w1)
                + self.tokenizer.img_token
                + imgstr1
                + self.tokenizer.eol_token
                + self.tokenizer.eof_token
                + self.tokenizer.eoi_token
            )

            image_prompt2 = (
                self.tokenizer.boi_token
                + self.processor.prefix_template.format(H=h2, W=w2)
                + self.tokenizer.img_token
                + imgstr2
                + self.tokenizer.eol_token
                + self.tokenizer.eof_token
                + self.tokenizer.eoi_token
            )

            # Build full prompt
            full_prompt = (
                self.tokenizer.bos_token
                + self.processor.chat_template.format(
                    image_prompt=image_prompt1 + " " + image_prompt2,
                    text_prompt=prompt_with_images,
                )
            )

            # Tokenize
            inputs = self.tokenizer(
                full_prompt,
                return_tensors="pt",
                padding="longest",
            )

            device = self.model.device
            input_ids = inputs.input_ids.to(device)
            attention_mask = inputs.attention_mask.to(device)

            # Generate answer
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.stage2_max_new_tokens,
                    do_sample=self.stage2_do_sample,
                    temperature=self.stage2_temperature if self.stage2_do_sample else None,
                    top_p=self.stage2_top_p if self.stage2_do_sample else None,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,  # Enable KV cache for faster generation
                )

            # Decode
            generated_text = self.processor.batch_decode(
                output_ids[:, input_ids.shape[1]:], skip_special_tokens=True
            )[0]

            # Clean up
            del inputs, input_ids, attention_mask, output_ids
            torch.cuda.empty_cache()

            eval_logger.debug(f"Stage 2 - Generated answer: {generated_text[:100]}...")
            return generated_text

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

        Stage 1: Generate auxiliary image from original image + text prompt
        Stage 2: Answer question using original image + auxiliary image + text
        """
        res = []

        pbar = tqdm(
            total=len(requests),
            desc="Emu3VisualCoT Generating",
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
                        original_visuals = self.flatten(original_visuals)
                        if len(original_visuals) > 0:
                            original_image = original_visuals[0]
                            if isinstance(original_image, str):
                                original_image = Image.open(original_image)
                            original_image = original_image.convert("RGB")
                            eval_logger.debug(
                                f"Extracted original image for doc {doc_id}"
                            )
                except Exception as e:
                    eval_logger.warning(
                        f"Failed to extract original image for doc {doc_id}: {e}"
                    )

            # Check if we have an original image (required for visual CoT)
            if original_image is None:
                eval_logger.warning(
                    f"No original image for doc {doc_id}, skipping visual CoT"
                )
                res.append("")
                pbar.update(1)
                continue

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
                stage2_contexts = contexts.replace(
                    f"[GEN_PROMPT]{gen_prompt_match.group(1)}[/GEN_PROMPT]", ""
                )
                stage2_contexts = stage2_contexts.replace(
                    f"[QUESTION]{question_match.group(1)}[/QUESTION]",
                    question_match.group(1),
                )
                eval_logger.info("Using custom generation prompt from task config")
            else:
                generation_prompt = self.generation_prompt_template.format(
                    question=contexts
                )
                stage2_contexts = contexts

            eval_logger.info(f"\n{'='*60}")
            eval_logger.info(f"Processing doc {doc_id} from task {task}")
            eval_logger.info(f"{'='*60}")

            # Stage 1: Generate auxiliary image
            stage1_text, generated_images = self._stage1_generate_image(
                generation_prompt=generation_prompt,
                original_image=original_image,
                doc_id=str(doc_id),
                task=task,
            )

            # Check if image was generated
            if not generated_images or len(generated_images) == 0:
                eval_logger.warning(
                    f"No auxiliary image generated for doc {doc_id}, returning empty"
                )
                res.append("")
                pbar.update(1)
                continue

            # Stage 2: Answer question using both images
            final_answer = self._stage2_answer_with_images(
                question=stage2_contexts,
                original_image=original_image,
                auxiliary_image_path=generated_images[0],
                doc_id=str(doc_id),
            )

            # Save intermediate artifacts if enabled
            self._save_intermediate_artifacts(
                doc_id=str(doc_id),
                task=task,
                generation_prompt=generation_prompt,
                stage1_text=stage1_text,
                generated_images=generated_images,
                question=stage2_contexts,
                stage2_answer=final_answer,
            )

            res.append(final_answer)
            pbar.update(1)

        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """Not supported for generation models"""
        raise NotImplementedError(
            "Emu3VisualCoT is a generation model and does not support loglikelihood"
        )

    def generate_until_multi_round(self, requests: List[Instance]) -> List[str]:
        """Multi-round dialogue not yet implemented"""
        raise NotImplementedError(
            "Multi-round dialogue not yet implemented for Emu3VisualCoT"
        )
