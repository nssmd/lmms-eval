"""
Emu3 Multimodal Model Integration

Emu3 is a multimodal LLM that uses vector quantization to tokenize images into discrete tokens.
It supports both text generation from images and image generation from text using next-token prediction.

Paper: https://huggingface.co/papers/2409.18869
Model: https://huggingface.co/BAAI/Emu3-Chat

Usage for understanding:
    python -m lmms_eval \
        --model emu3 \
        --model_args pretrained=BAAI/Emu3-Chat,mode=understanding \
        --tasks mmbench \
        --batch_size 1 \
        --output_path ./logs/

Note: This model requires trust_remote_code=True and uses a separate vision tokenizer
(BAAI/Emu3-VisionTokenizer) for image encoding.
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


@register_model("emu3")
class Emu3(lmms):
    """
    Emu3: Next-Token Prediction Multimodal Model

    Supports:
    - Text generation from images (understanding mode)
    - Image generation from text (generation mode)

    Both modes use the same Emu3-Chat model since it's a unified architecture.
    Requires BAAI/Emu3-VisionTokenizer for image encoding.
    """

    def __init__(
        self,
        pretrained: str = "BAAI/Emu3-Chat",
        vq_hub: str = "BAAI/Emu3-VisionTokenizer",
        mode: str = "understanding",
        device: str = "cuda",
        device_map: str = "auto",
        batch_size: int = 1,
        # Text generation parameters
        max_new_tokens: int = 512,
        do_sample: bool = False,
        temperature: float = 0.0,
        top_p: float = 1.0,
        # Image generation parameters
        image_generation_max_tokens: int = 50000,
        image_height: int = 512,
        image_width: int = 512,
        image_ratio: str = "1:1",
        negative_prompt: str = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry.",
        # Model loading
        use_flash_attention_2: bool = True,
        torch_dtype: str = "bfloat16",
        # Output
        output_image_dir: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        # Validate mode
        if mode not in ["understanding", "generation"]:
            raise ValueError(
                f"mode must be 'understanding' or 'generation', got '{mode}'"
            )

        self.pretrained = pretrained
        self.vq_hub = vq_hub
        self.mode = mode
        self.device_str = device
        self.device_map = device_map
        self.batch_size_per_gpu = batch_size

        # Text generation parameters
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p

        # Image generation parameters
        self.image_generation_max_tokens = image_generation_max_tokens
        self.negative_prompt = negative_prompt

        # Set image dimensions based on ratio
        if image_ratio == "1:1":
            self.image_height = 512
            self.image_width = 512
        elif image_ratio == "4:3":
            self.image_height = 384
            self.image_width = 512
        elif image_ratio == "3:4":
            self.image_height = 512
            self.image_width = 384
        elif image_ratio == "16:9":
            self.image_height = 288
            self.image_width = 512
        elif image_ratio == "9:16":
            self.image_height = 512
            self.image_width = 288
        else:
            self.image_height = image_height
            self.image_width = image_width

        self.use_flash_attention_2 = use_flash_attention_2

        # Set torch dtype
        if torch_dtype == "bfloat16":
            self.torch_dtype = torch.bfloat16
        elif torch_dtype == "float16":
            self.torch_dtype = torch.float16
        else:
            self.torch_dtype = torch.float32

        # Setup output directory for generated images
        if output_image_dir is None:
            self.output_image_dir = "./logs/emu3_generated_images"
        else:
            self.output_image_dir = output_image_dir

        if self.mode == "generation":
            os.makedirs(self.output_image_dir, exist_ok=True)
            eval_logger.info(f"Image output directory: {self.output_image_dir}")

        # Load model and processor
        eval_logger.info(f"Loading Emu3 model from {pretrained} in {mode} mode")
        self._load_model()
        eval_logger.info("Emu3 initialized successfully")

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

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    def flatten(self, input_list):
        """Flatten nested lists"""
        output = []
        for item in input_list:
            if isinstance(item, list):
                output.extend(self.flatten(item))
            else:
                output.append(item)
        return output

    def format_output(self, text: str, images: List[str]) -> str:
        """Format output as JSON string"""
        output_dict = {"text": text, "images": images}
        return json.dumps(output_dict, ensure_ascii=False)

    def _resize_image(self, image: Image.Image, max_size: int = 512) -> Image.Image:
        """Resize image to max_size x max_size to save memory"""
        if image.width > max_size or image.height > max_size:
            image = image.resize((max_size, max_size), Image.Resampling.LANCZOS)
        return image
        return image

    def _generate_text_from_image(
        self, context: str, images: List[Image.Image], gen_kwargs: dict
    ) -> str:
        """Generate text response from image(s) - understanding mode"""
        # For single image, we can pass it directly
        if len(images) == 1:
            image = images[0]
        else:
            eval_logger.warning(
                f"Multiple images provided ({len(images)}), using first image only"
            )
            image = images[0]

        # Ensure image is RGB
        if hasattr(image, "mode") and image.mode != "RGB":
            image = image.convert("RGB")

        # Resize image to 512x512 to save memory
        image = self._resize_image(image, max_size=512)

        # Prepare inputs using custom processor with mode='U' for understanding
        inputs = self.processor(
            text=context,
            image=image,
            mode="U",  # 'U' mode for understanding/perception tasks
            return_tensors="pt",
            padding="longest",
        )

        # Move to device
        device = self.model.device
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)

        # Get generation parameters
        max_new_tokens = gen_kwargs.get("max_new_tokens", self.max_new_tokens)
        do_sample = gen_kwargs.get("do_sample", self.do_sample)
        temperature = gen_kwargs.get("temperature", self.temperature)
        top_p = gen_kwargs.get("top_p", self.top_p)

        # Generate using simple generate call without GenerationConfig
        # to avoid DynamicCache compatibility issues
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=False,  # Disable cache to avoid DynamicCache issues
            )

        # Decode - skip the input tokens
        generated_text = self.processor.batch_decode(
            output_ids[:, input_ids.shape[1]:], skip_special_tokens=True
        )[0]

        # Clean up to free memory
        del inputs, input_ids, attention_mask, output_ids
        torch.cuda.empty_cache()

        return generated_text

    def _generate_text_only(self, context: str, gen_kwargs: dict) -> str:
        """Generate text-only response (no image input)"""
        # Prepare inputs without image
        inputs = self.processor(
            text=context,
            mode="U",
            return_tensors="pt",
            padding="longest",
        )

        # Move to device
        device = self.model.device
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)

        # Get generation parameters
        max_new_tokens = gen_kwargs.get("max_new_tokens", self.max_new_tokens)
        do_sample = gen_kwargs.get("do_sample", self.do_sample)
        temperature = gen_kwargs.get("temperature", self.temperature)
        top_p = gen_kwargs.get("top_p", self.top_p)

        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=False,
            )

        # Decode
        generated_text = self.processor.batch_decode(
            output_ids[:, input_ids.shape[1]:], skip_special_tokens=True
        )[0]

        return generated_text

    def _generate_image(
        self,
        prompt: str,
        doc_id: str,
        task: str,
        gen_kwargs: dict,
    ) -> Tuple[str, List[str]]:
        """Generate image from text prompt - generation mode"""
        # Prepare inputs for image generation with mode='G'
        inputs = self.processor(
            text=prompt,
            mode="G",  # 'G' mode for generation
            ratio=f"{self.image_width}:{self.image_height}",
            return_tensors="pt",
            padding="longest",
        )

        # Move to device
        device = self.model.device
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)

        # Prepare negative prompt
        neg_inputs = self.processor(
            text=self.negative_prompt,
            mode="G",
            ratio=f"{self.image_width}:{self.image_height}",
            return_tensors="pt",
        )
        neg_input_ids = neg_inputs.input_ids.to(device)
        neg_attention_mask = neg_inputs.attention_mask.to(device)

        # Get image generation parameters
        HEIGHT, WIDTH = self.image_height, self.image_width

        # Define prefix constraint function for image generation
        def prefix_allowed_tokens_fn(batch_id, input_ids_seq):
            height, width = HEIGHT, WIDTH
            visual_tokens = self.processor.visual_tokens

            image_wrapper_token_id = torch.tensor(
                [self.tokenizer.image_wrapper_token_id],
                device=device,
            )
            eoi_token_id = torch.tensor(
                [self.tokenizer.eoi_token_id], device=device
            )
            eos_token_id = torch.tensor(
                [self.tokenizer.eos_token_id], device=device
            )
            pad_token_id = torch.tensor(
                [self.tokenizer.pad_token_id], device=device
            )
            eof_token_id = torch.tensor(
                [self.tokenizer.eof_token_id], device=device
            )
            eol_token_id = self.tokenizer.encode(
                "<|extra_200|>", return_tensors="pt"
            )[0]

            position = torch.nonzero(
                input_ids_seq == image_wrapper_token_id, as_tuple=True
            )[0][0]
            offset = input_ids_seq.shape[0] - position
            if offset % (width + 1) == 0:
                return (eol_token_id,)
            elif offset == (width + 1) * height + 1:
                return (eof_token_id,)
            elif offset == (width + 1) * height + 2:
                return (eoi_token_id,)
            elif offset == (width + 1) * height + 3:
                return (eos_token_id,)
            elif offset > (width + 1) * height + 3:
                return (pad_token_id,)
            else:
                return visual_tokens

        # Generate
        with torch.no_grad():
            out = self.model.generate(
                input_ids,
                max_new_tokens=self.image_generation_max_tokens,
                attention_mask=attention_mask,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                return_dict_in_generate=True,
                negative_prompt_ids=neg_input_ids,
                negative_prompt_attention_mask=neg_attention_mask,
            )

        # Decode image tokens using processor
        images = self.processor.decode_image_tokens(
            out.sequences[:, input_ids.shape[1]:],
            height=HEIGHT,
            width=WIDTH,
        )

        # Save generated images
        output_images = []
        for i, img in enumerate(images):
            safe_filename = f"{task}_{doc_id}_{i}.png"
            image_path = os.path.join(self.output_image_dir, safe_filename)
            img.save(image_path)
            output_images.append(image_path)
            eval_logger.info(f"Saved generated image: {image_path}")

        # Return empty text and image paths
        return "", output_images

    def generate_until(self, requests: List[Instance]) -> List[str]:
        """Generate responses for a list of requests"""
        res = []

        pbar = tqdm(
            total=len(requests),
            desc=f"Generating with Emu3 ({self.mode})",
        )

        for request in requests:
            contexts, gen_kwargs, doc_to_visual, doc_id, task, split = request.args

            # Get images if available
            images = []
            if doc_to_visual is not None:
                doc = self.task_dict[task][split][doc_id]
                visuals = [doc_to_visual(doc)]
                visuals = self.flatten(visuals)

                for visual in visuals:
                    if isinstance(visual, str):
                        # Path to image
                        img = Image.open(visual).convert("RGB")
                    elif isinstance(visual, Image.Image):
                        img = visual.convert("RGB")
                    else:
                        eval_logger.warning(f"Unsupported visual type: {type(visual)}")
                        continue
                    images.append(img)

            # Debug: log image info
            if images:
                eval_logger.debug(
                    f"doc_id={doc_id}, images loaded: {len(images)}, "
                    f"size={images[0].size if images else 'N/A'}"
                )

            # Generate response based on mode
            if self.mode == "understanding":
                # Understanding mode: image + text -> text
                if len(images) > 0:
                    response = self._generate_text_from_image(
                        contexts, images, gen_kwargs
                    )
                else:
                    response = self._generate_text_only(contexts, gen_kwargs)
                formatted_output = response
            else:
                # Generation mode: text -> image
                output_text, output_images = self._generate_image(
                    contexts, str(doc_id), task, gen_kwargs
                )
                formatted_output = self.format_output(output_text, output_images)

            res.append(formatted_output)
            pbar.update(1)

        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """Calculate log-likelihood for requests"""
        eval_logger.warning(
            "loglikelihood not implemented for Emu3, returning dummy values"
        )
        return [(0.0, False) for _ in requests]

    def generate_until_multi_round(self, requests: List[Instance]) -> List[str]:
        """Generate responses for multi-round conversations"""
        eval_logger.warning(
            "generate_until_multi_round not fully implemented for Emu3, "
            "using generate_until"
        )
        return self.generate_until(requests)
