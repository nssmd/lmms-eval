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
        Get visual tokens from the tokenizer vocabulary.

        Emu3 uses VQ-VAE to tokenize images into discrete tokens.
        The visual tokens are typically in a specific range of the vocabulary.
        """
        # Try to get visual_tokens from processor first
        if hasattr(self.processor, "visual_tokens"):
            return self.processor.visual_tokens

        # Try to get from image_tokenizer's codebook size
        if hasattr(self.image_tokenizer, "config"):
            codebook_size = getattr(
                self.image_tokenizer.config, "codebook_size", 32768
            )
        else:
            codebook_size = 32768  # Default Emu3 codebook size

        # Find visual tokens in tokenizer vocabulary
        # Emu3 visual tokens typically start after special tokens
        # and are in the format <|visual token XXXX|> or similar
        visual_token_ids = []

        # Method 1: Look for tokens with "visual" in the name
        vocab = self.tokenizer.get_vocab()
        for token, token_id in vocab.items():
            # Handle both string and bytes tokens
            if isinstance(token, bytes):
                token_str = token.decode("utf-8", errors="ignore")
            else:
                token_str = str(token)
            if "visual" in token_str.lower() or "image" in token_str.lower():
                if "token" in token_str.lower() and token_id not in visual_token_ids:
                    visual_token_ids.append(token_id)

        # Method 2: If no visual tokens found, use a range based on codebook size
        # Emu3 typically uses token IDs starting from a specific offset
        if len(visual_token_ids) < codebook_size:
            # Find the starting point for visual tokens
            # Usually after all text tokens and special tokens
            vocab_size = len(vocab)

            # Emu3 typically has visual tokens at the end of vocabulary
            # or in a specific range. Let's try to find them.
            # Common pattern: visual tokens are from (vocab_size - codebook_size) to vocab_size
            start_id = vocab_size - codebook_size
            if start_id > 0:
                visual_token_ids = list(range(start_id, vocab_size))
            else:
                # Fallback: use a reasonable range
                # Emu3-Chat typically has visual tokens from 151643 to 184410
                visual_token_ids = list(range(151643, 151643 + codebook_size))

        eval_logger.debug(
            f"Found {len(visual_token_ids)} visual tokens "
            f"(range: {min(visual_token_ids)}-{max(visual_token_ids)})"
        )

        return tuple(visual_token_ids)

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

            # Step 4: Define prefix constraint function for image generation
            # Use pre-computed visual tokens from self.visual_tokens
            visual_tokens = self.visual_tokens

            # Get special token IDs - handle different tokenizer attribute names
            def get_token_id(tokenizer, attr_name, fallback_token):
                """Get token ID from tokenizer attribute or by encoding the token string"""
                if hasattr(tokenizer, attr_name):
                    return getattr(tokenizer, attr_name)
                # Try to encode the fallback token string
                try:
                    ids = tokenizer.encode(fallback_token, add_special_tokens=False)
                    if ids:
                        return ids[0]
                except Exception:
                    pass
                return None

            # Debug: Log available tokenizer attributes
            special_attrs = [
                attr for attr in dir(self.tokenizer)
                if "token" in attr.lower() and not attr.startswith("_")
            ]
            eval_logger.debug(f"Tokenizer special token attrs: {special_attrs}")

            # Get token IDs with more fallbacks
            # Try multiple possible attribute names for the image wrapper token
            img_token_id = None
            for attr, fallback in [
                ("image_wrapper_token_id", "<|image token|>"),
                ("img_token_id", "<|img|>"),
                ("boi_token_id", "<|boi|>"),
            ]:
                img_token_id = get_token_id(self.tokenizer, attr, fallback)
                if img_token_id is not None:
                    eval_logger.debug(f"Found img_token_id via {attr}: {img_token_id}")
                    break

            # If still None, try to find it in the input_ids we just created
            if img_token_id is None:
                # The img_token should be in the generation prompt we added
                # Look for it by checking the tokenizer's special tokens
                if hasattr(self.tokenizer, "img_token"):
                    img_token_str = self.tokenizer.img_token
                    ids = self.tokenizer.encode(img_token_str, add_special_tokens=False)
                    if ids:
                        img_token_id = ids[0]
                        eval_logger.debug(
                            f"Found img_token_id via img_token attr: {img_token_id}"
                        )

            eval_logger.info(f"Image token ID: {img_token_id}")

            eoi_id = get_token_id(self.tokenizer, "eoi_token_id", "<|eoi|>")
            eof_id = get_token_id(self.tokenizer, "eof_token_id", "<|eof|>")
            eos_id = self.tokenizer.eos_token_id
            pad_id = self.tokenizer.pad_token_id

            eval_logger.debug(
                f"Special token IDs - eoi: {eoi_id}, eof: {eof_id}, "
                f"eos: {eos_id}, pad: {pad_id}"
            )

            # Get eol token ID
            eol_ids = self.tokenizer.encode("<|extra_200|>", add_special_tokens=False)
            eol_id = eol_ids[0] if eol_ids else None
            eval_logger.debug(f"EOL token ID: {eol_id}")

            # Store scalar values for comparison in prefix_allowed_tokens_fn
            # (tensors might have device mismatch issues)
            img_token_scalar = img_token_id
            eoi_scalar = eoi_id
            eof_scalar = eof_id
            eos_scalar = eos_id
            pad_scalar = pad_id
            eol_scalar = eol_id

            # Debug: track if prefix_allowed_tokens_fn is working
            debug_call_count = [0]
            debug_logged = [False]

            def prefix_allowed_tokens_fn(batch_id, input_ids_seq):
                height, width = h_tokens, w_tokens
                debug_call_count[0] += 1

                # If we don't have the image wrapper token, just return visual tokens
                if img_token_scalar is None:
                    if not debug_logged[0]:
                        eval_logger.warning("img_token_scalar is None!")
                        debug_logged[0] = True
                    return visual_tokens

                # Move input_ids_seq to CPU for comparison if needed
                if input_ids_seq.device.type != "cpu":
                    input_ids_cpu = input_ids_seq.cpu()
                else:
                    input_ids_cpu = input_ids_seq

                # Find the last image wrapper token (for the generated image)
                positions = (input_ids_cpu == img_token_scalar).nonzero(as_tuple=True)[0]

                # Debug logging (only first few calls)
                if debug_call_count[0] <= 3:
                    eval_logger.info(
                        f"prefix_fn call {debug_call_count[0]}: "
                        f"seq_len={len(input_ids_cpu)}, "
                        f"positions found={len(positions)}, "
                        f"looking for token {img_token_scalar}"
                    )

                if len(positions) == 0:
                    if not debug_logged[0]:
                        eval_logger.warning(
                            f"img_token {img_token_scalar} not found in sequence! "
                            f"Seq length: {len(input_ids_cpu)}, "
                            f"Last 10 tokens: {input_ids_cpu[-10:].tolist()}"
                        )
                        debug_logged[0] = True
                    return visual_tokens

                position = positions[-1].item()  # Use last position for generated image
                offset = len(input_ids_cpu) - position

                # Debug: log offset periodically
                if debug_call_count[0] <= 5 or debug_call_count[0] % 100 == 0:
                    eval_logger.debug(
                        f"offset={offset}, expected_end={(width + 1) * height + 3}"
                    )

                if offset % (width + 1) == 0 and eol_scalar is not None:
                    return [eol_scalar]
                elif offset == (width + 1) * height + 1 and eof_scalar is not None:
                    return [eof_scalar]
                elif offset == (width + 1) * height + 2 and eoi_scalar is not None:
                    return [eoi_scalar]
                elif offset == (width + 1) * height + 3 and eos_scalar is not None:
                    return [eos_scalar]
                elif offset > (width + 1) * height + 3 and pad_scalar is not None:
                    return [pad_scalar]
                else:
                    return visual_tokens

            # Step 5: Generate image tokens with progress tracking
            # Calculate expected number of tokens for progress bar
            # Total = h_tokens * (w_tokens + 1) + 3 (visual + EOL + EOF/EOI/EOS)
            expected_tokens = h_tokens * (w_tokens + 1) + 3

            # Create a progress bar for image generation
            gen_pbar = tqdm(
                total=expected_tokens,
                desc=f"  Stage1 Image Gen (doc {doc_id})",
                leave=False,
                unit="tok",
            )

            # Custom streamer to track progress
            class ProgressStreamer:
                def __init__(self, pbar):
                    self.pbar = pbar
                    self.token_count = 0

                def put(self, value):
                    # value is a tensor of token IDs
                    if hasattr(value, "shape"):
                        new_tokens = value.shape[-1]
                    else:
                        new_tokens = 1
                    self.token_count += new_tokens
                    self.pbar.update(new_tokens)

                def end(self):
                    self.pbar.close()

            streamer = ProgressStreamer(gen_pbar)

            with torch.no_grad():
                out = self.model.generate(
                    input_ids,
                    max_new_tokens=self.stage1_max_new_tokens,
                    attention_mask=attention_mask,
                    prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                    return_dict_in_generate=True,
                    negative_prompt_ids=neg_input_ids,
                    negative_prompt_attention_mask=neg_attention_mask,
                    use_cache=False,  # Disable cache to avoid DynamicCache issues
                    streamer=streamer,
                )

            gen_pbar.close()

            # Step 6: Decode image tokens
            images = self.processor.decode_image_tokens(
                out.sequences[:, input_length:],
                height=h_tokens,
                width=w_tokens,
            )

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
                    use_cache=False,
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
