"""
UAE Model - Unified Multimodal Model as Auto-Encoder

UAE is a unified multimodal model that combines:
- Encoder: Qwen-2.5-VL-3B (LVLM for image understanding)
- Projector: MLP (maps hidden states to SD3 conditioning space)
- Decoder: SD3.5-Large (DiT for image generation)

Paper: https://arxiv.org/abs/2509.09666
GitHub: https://github.com/PKU-YuanGroup/UAE
HuggingFace: https://huggingface.co/zhiyuanyan1/UAE

Modes:
    - "understanding": Visual understanding (image + text -> text)
    - "generation": Image generation (text -> image) or (image + text -> image)

Example usage for understanding:
    python -m lmms_eval --model uae \
        --model_args pretrained=zhiyuanyan1/UAE,mode=understanding \
        --tasks mmbench --batch_size 1 --device cuda:0

Example usage for generation:
    python -m lmms_eval --model uae \
        --model_args pretrained=zhiyuanyan1/UAE,mode=generation \
        --tasks geneval --batch_size 1 --device cuda:0
"""

import json
import os
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model


@register_model("uae")
class UAE(lmms):
    """
    UAE Model - Unified Multimodal Model as Auto-Encoder

    Architecture:
        - Encoder: Qwen-2.5-VL-3B with LoRA
        - Projector: MLP (hidden_size -> 4096*2 -> 4096)
        - Decoder: SD3.5-Large with LoRA
    """

    def __init__(
        self,
        pretrained: str = "zhiyuanyan1/UAE",
        mode: str = "understanding",
        device: str = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = torch.bfloat16,
        batch_size: int = 1,
        # LLM (Encoder) parameters
        llm_max_new_tokens: int = 512,
        llm_temperature: float = 0.7,
        llm_do_sample: bool = True,
        # SD3 (Decoder) parameters
        sd3_num_inference_steps: int = 40,
        sd3_guidance_scale: float = 5.0,
        sd3_height: int = 1024,
        sd3_width: int = 1024,
        # LoRA parameters
        lora_rank: int = 32,
        lora_alpha: int = 64,
        # Output settings
        output_image_dir: Optional[str] = None,
        response_persistent_folder: Optional[str] = None,
        continual_mode: bool = True,
        # Memory optimization
        offload_encoder_after_use: bool = False,
        seed: int = 42,
        **kwargs,
    ) -> None:
        super().__init__()

        # Validate mode
        if mode not in ["understanding", "generation"]:
            raise ValueError(
                f"mode must be 'understanding' or 'generation', got '{mode}'"
            )

        self.pretrained = pretrained
        self.mode = mode
        self.batch_size_per_gpu = int(batch_size)
        self.seed = seed
        self.offload_encoder_after_use = offload_encoder_after_use

        # LLM parameters
        self.llm_max_new_tokens = llm_max_new_tokens
        self.llm_temperature = llm_temperature
        self.llm_do_sample = llm_do_sample

        # SD3 parameters
        self.sd3_num_inference_steps = sd3_num_inference_steps
        self.sd3_guidance_scale = sd3_guidance_scale
        self.sd3_height = sd3_height
        self.sd3_width = sd3_width

        # LoRA parameters
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha

        # Set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Determine device
        if torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            if world_size > 1:
                self._device = torch.device(f"cuda:{local_rank}")
            else:
                self._device = (
                    device if isinstance(device, torch.device) else torch.device(device)
                )
        else:
            self._device = torch.device("cpu")

        # Determine dtype
        if dtype == "auto" or dtype is None:
            self._dtype = torch.bfloat16
        elif isinstance(dtype, str):
            dtype_map = {
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }
            self._dtype = dtype_map.get(dtype, torch.bfloat16)
        else:
            self._dtype = dtype

        # Setup output directories
        if response_persistent_folder is None:
            self.response_persistent_folder = "./logs/uae_persistent_folder"
        else:
            self.response_persistent_folder = response_persistent_folder

        if output_image_dir is None:
            self.output_image_dir = os.path.join(
                self.response_persistent_folder, "uae_generated_images"
            )
        else:
            self.output_image_dir = output_image_dir

        os.makedirs(self.output_image_dir, exist_ok=True)
        eval_logger.info(f"Image output directory: {self.output_image_dir}")

        # Setup response cache for continual mode
        self.response_cache = {}
        self.cache_mode = "start"
        self.continual_mode = continual_mode

        if self.continual_mode:
            os.makedirs(self.response_persistent_folder, exist_ok=True)
            self.response_persistent_file = os.path.join(
                self.response_persistent_folder, "uae_response.json"
            )
            if os.path.exists(self.response_persistent_file):
                with open(self.response_persistent_file, "r") as f:
                    self.response_cache = json.load(f)
                self.cache_mode = "resume"
                eval_logger.info(f"Loaded cache: {len(self.response_cache)} records")

        # Setup distributed training
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        if world_size > 1:
            from accelerate import Accelerator

            accelerator = Accelerator()
            self.accelerator = accelerator
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
            if self.continual_mode:
                eval_logger.warning(
                    "Continual mode is not supported for distributed inference. "
                    "Automatically disabling continual_mode."
                )
                self.continual_mode = False
        else:
            self._rank = 0
            self._world_size = 1

        # Load models
        eval_logger.info(f"Loading UAE model from {pretrained}")
        self._load_models()

        eval_logger.info("UAE model initialized successfully")

    def _load_models(self):
        """Load UAE model components based on mode"""
        # Always load the encoder (Qwen-2.5-VL) for understanding
        self._load_encoder()

        # Load decoder (SD3.5) only for generation mode
        if self.mode == "generation":
            self._load_decoder()

    def _load_encoder(self):
        """Load Qwen-2.5-VL encoder with LoRA and Projector"""
        eval_logger.info("Loading Qwen-2.5-VL encoder...")

        try:
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
            from peft import LoraConfig, PeftModel
            from safetensors.torch import load_file
        except ImportError as e:
            raise ImportError(
                f"UAE requires transformers and peft packages. Error: {e}"
            )

        # Determine paths
        if os.path.isdir(self.pretrained):
            # Local path
            llm_path = os.path.join(self.pretrained, "llm_model")
            llm_lora_path = os.path.join(self.pretrained, "llm_lora")
            processor_path = os.path.join(self.pretrained, "llm_model")
        else:
            # HuggingFace path - download components
            from huggingface_hub import snapshot_download

            cache_dir = snapshot_download(
                repo_id=self.pretrained,
                allow_patterns=["llm_model/*", "llm_lora/*"],
            )
            llm_path = os.path.join(cache_dir, "llm_model")
            llm_lora_path = os.path.join(cache_dir, "llm_lora")
            processor_path = llm_path

            # Check if merges.txt exists, if not use Qwen2.5-VL-3B-Instruct as fallback
            merges_file = os.path.join(processor_path, "merges.txt")
            if not os.path.exists(merges_file):
                eval_logger.warning(
                    f"merges.txt not found in {processor_path}, "
                    "using Qwen/Qwen2.5-VL-3B-Instruct as processor fallback"
                )
                processor_path = "Qwen/Qwen2.5-VL-3B-Instruct"

        # Define custom model class with projector (matching UAE's architecture)
        class Qwen2_5_VLWithProjector(Qwen2_5_VLForConditionalGeneration):
            """Qwen2.5-VL with projector for SD3 integration"""
            def __init__(self, config):
                super().__init__(config)
                # Qwen2.5-VL-3B hidden_size is 2048
                hidden_size = getattr(config, 'hidden_size', 2048)
                self.projector = nn.Sequential(
                    nn.Linear(hidden_size, 4096 * 2),
                    nn.GELU(),
                    nn.Linear(4096 * 2, 4096),
                )

            def get_projected_embeddings(
                self,
                input_ids=None,
                attention_mask=None,
                pixel_values=None,
                image_grid_thw=None,
                **kwargs
            ):
                """Get embeddings with projection applied for SD3"""
                outputs = super().forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    output_hidden_states=True,
                    return_dict=True,
                    **kwargs
                )
                hidden_states = outputs.hidden_states[-1]
                projected = self.projector(hidden_states)
                return projected

        # Load base model with projector
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(llm_path, trust_remote_code=True)

        self._encoder = Qwen2_5_VLWithProjector(config)

        # Load weights (including projector weights)
        import glob
        safetensor_files = sorted(glob.glob(os.path.join(llm_path, "*.safetensors")))

        state_dict = {}
        for f in safetensor_files:
            weights = load_file(f)
            state_dict.update(weights)

        # Load state dict
        missing, unexpected = self._encoder.load_state_dict(state_dict, strict=False)
        if missing:
            eval_logger.warning(f"Missing keys when loading encoder: {missing[:5]}...")
        if unexpected:
            eval_logger.warning(f"Unexpected keys when loading encoder: {unexpected[:5]}...")

        self._encoder = self._encoder.to(self._device, dtype=self._dtype)
        eval_logger.info("Loaded encoder with projector")

        # Load LoRA weights
        if os.path.exists(llm_lora_path):
            lora_state_dict = {}
            for i in range(1, 4):
                safe_tensor_path = os.path.join(
                    llm_lora_path, f"model-0000{i}-of-00003.safetensors"
                )
                if os.path.exists(safe_tensor_path):
                    weights = load_file(safe_tensor_path)
                    lora_state_dict.update(weights)

            if lora_state_dict:
                self._encoder.load_state_dict(lora_state_dict, strict=False)
                eval_logger.info("Loaded LLM LoRA weights")

        # Load processor
        self._processor = AutoProcessor.from_pretrained(
            processor_path, trust_remote_code=True
        )
        self._tokenizer = self._processor.tokenizer

        self._encoder.eval()
        eval_logger.info("Encoder loaded successfully")

    def _load_decoder(self):
        """Load SD3.5-Large decoder with LoRA"""
        eval_logger.info("Loading SD3.5-Large decoder...")

        try:
            from diffusers import (
                StableDiffusion3Pipeline,
                SD3Transformer2DModel,
                AutoencoderKL,
            )
            from transformers import (
                CLIPTextModelWithProjection,
                T5EncoderModel,
                CLIPTokenizer,
                T5TokenizerFast,
            )
            from peft import PeftModel
        except ImportError as e:
            raise ImportError(f"UAE generation mode requires diffusers. Error: {e}")

        # Determine paths
        if os.path.isdir(self.pretrained):
            sd3_path = os.path.join(self.pretrained, "SD3")
            dit_path = os.path.join(self.pretrained, "dit")
            dit_lora_path = os.path.join(self.pretrained, "dit_lora")
        else:
            from huggingface_hub import snapshot_download

            cache_dir = snapshot_download(
                repo_id=self.pretrained,
                allow_patterns=["SD3/*", "dit/*", "dit_lora/*"],
            )
            sd3_path = os.path.join(cache_dir, "SD3")
            dit_path = os.path.join(cache_dir, "dit")
            dit_lora_path = os.path.join(cache_dir, "dit_lora")

        # Load transformer
        transformer = SD3Transformer2DModel.from_pretrained(
            dit_path,
            torch_dtype=self._dtype,
        ).to(self._device)

        # Load VAE
        vae = AutoencoderKL.from_pretrained(
            sd3_path,
            subfolder="vae",
            torch_dtype=self._dtype,
        ).to(self._device)

        # Load text encoders for pooled embeddings
        self._text_encoder_1 = CLIPTextModelWithProjection.from_pretrained(
            sd3_path,
            torch_dtype=self._dtype,
            subfolder="text_encoder",
        ).to(self._device)
        self._text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            sd3_path,
            torch_dtype=self._dtype,
            subfolder="text_encoder_2",
        ).to(self._device)
        self._text_encoder_3 = T5EncoderModel.from_pretrained(
            sd3_path,
            torch_dtype=self._dtype,
            subfolder="text_encoder_3",
        ).to(self._device)

        # Load tokenizers
        self._tokenizer_1 = CLIPTokenizer.from_pretrained(
            sd3_path, subfolder="tokenizer"
        )
        self._tokenizer_2 = CLIPTokenizer.from_pretrained(
            sd3_path, subfolder="tokenizer_2"
        )
        self._tokenizer_3 = T5TokenizerFast.from_pretrained(
            sd3_path, subfolder="tokenizer_3"
        )

        # Create text-to-image pipeline
        self._pipe = StableDiffusion3Pipeline.from_pretrained(
            sd3_path,
            vae=vae,
            transformer=transformer,
            text_encoder=None,
            text_encoder_2=None,
            text_encoder_3=None,
            torch_dtype=self._dtype,
        ).to(self._device)

        # Apply LoRA to transformer
        if os.path.exists(dit_lora_path):
            self._pipe.transformer = PeftModel.from_pretrained(
                self._pipe.transformer, dit_lora_path
            )
            eval_logger.info("Loaded DiT LoRA weights")

        # Pre-compute pooled embeddings for empty prompt
        self._pooled_prompt_embeds = self._get_pooled_embeds("")

        eval_logger.info("Decoder loaded successfully")

    def _get_pooled_embeds(self, prompt: str) -> torch.Tensor:
        """Get pooled prompt embeddings from CLIP encoders"""
        # Tokenize
        text_input_1 = self._tokenizer_1(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        ).to(self._device)

        text_input_2 = self._tokenizer_2(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        ).to(self._device)

        # Get pooled outputs
        with torch.no_grad():
            pooled_1 = self._text_encoder_1(
                text_input_1.input_ids, output_hidden_states=False
            ).text_embeds
            pooled_2 = self._text_encoder_2(
                text_input_2.input_ids, output_hidden_states=False
            ).text_embeds

        # Concatenate pooled embeddings
        pooled = torch.cat([pooled_1, pooled_2], dim=-1)
        return pooled

    def _get_prompt_embeds(
        self, prompt: str, image: Optional[Image.Image] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get prompt embeddings using Qwen encoder + projector

        If image is provided, Qwen will encode both image and text together,
        producing embeddings that contain visual + textual semantics.

        Args:
            prompt: Text prompt for generation
            image: Optional input image to condition on

        Returns:
            Tuple of (negative_prompt_embeds, positive_prompt_embeds)
        """
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError:
            process_vision_info = None

        def get_embeds(text: str, img: Optional[Image.Image] = None) -> torch.Tensor:
            """Get embeddings for text (optionally with image)"""
            if img is not None:
                # Encode image + text together
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img},
                            {"type": "text", "text": text},
                        ],
                    }
                ]
            else:
                # Text only
                messages = [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": text}],
                    }
                ]

            # Apply chat template
            formatted_text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Process vision info if image provided
            if img is not None and process_vision_info is not None:
                image_inputs, video_inputs = process_vision_info(messages)
            else:
                image_inputs = [img] if img is not None else None
                video_inputs = None

            # Prepare model inputs
            model_inputs = self._processor(
                text=[formatted_text],
                images=image_inputs,
                videos=video_inputs,
                return_tensors="pt",
                padding=True,
            ).to(self._device)

            # Get projected embeddings from Qwen
            with torch.no_grad():
                embeds = self._encoder.get_projected_embeddings(**model_inputs)

            return embeds

        # Get negative prompt embeddings (text only, no image)
        negative_prompt = (
            "Generate a random, low quality, ugly, blur, bad and anime, cartoon image."
        )
        negative_embeds = get_embeds(negative_prompt, img=None)

        # Get positive prompt embeddings (with image if provided)
        positive_embeds = get_embeds(prompt, img=image)

        return negative_embeds, positive_embeds

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    @property
    def model(self):
        return self._encoder

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    def flatten(self, input_list: List) -> List:
        """Flatten a nested list"""
        output = []
        for item in input_list:
            if isinstance(item, list):
                output.extend(self.flatten(item))
            else:
                output.append(item)
        return output

    def _extract_image(self, visual) -> Optional[Image.Image]:
        """Extract PIL Image from various formats"""
        try:
            if visual is None:
                return None
            elif isinstance(visual, Image.Image):
                return visual.convert("RGB")
            elif isinstance(visual, str):
                return Image.open(visual).convert("RGB")
            elif isinstance(visual, dict):
                if "bytes" in visual:
                    from io import BytesIO

                    return Image.open(BytesIO(visual["bytes"])).convert("RGB")
                elif "path" in visual:
                    return Image.open(visual["path"]).convert("RGB")
                elif "image" in visual:
                    return self._extract_image(visual["image"])
            return None
        except Exception as e:
            eval_logger.warning(f"Failed to extract image: {e}")
            return None

    def understand_image(
        self, prompt: str, image: Optional[Image.Image], doc_id: str
    ) -> str:
        """
        Understand image and answer question using Qwen-2.5-VL

        Args:
            prompt: Input text prompt/question
            image: PIL Image to understand (optional)
            doc_id: Document ID for logging

        Returns:
            Generated text answer
        """
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError:
            eval_logger.warning(
                "qwen_vl_utils not found, using basic image processing"
            )
            process_vision_info = None

        # Prepare messages
        if image is not None:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
        else:
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}],
                }
            ]

        # Apply chat template
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Process vision info
        if process_vision_info is not None and image is not None:
            image_inputs, video_inputs = process_vision_info(messages)
        else:
            image_inputs = [image] if image is not None else None
            video_inputs = None

        # Prepare model inputs
        model_inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
        ).to(self._device)

        # Generate
        with torch.no_grad():
            generated_ids = self._encoder.generate(
                **model_inputs,
                max_new_tokens=self.llm_max_new_tokens,
                do_sample=self.llm_do_sample,
                temperature=self.llm_temperature,
            )

        # Decode
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self._processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        return response

    def generate_image(
        self,
        prompt: str,
        doc_id: str,
        task: str,
        image: Optional[Image.Image] = None,
    ) -> Tuple[str, List[str]]:
        """
        Generate image from text prompt (optionally conditioned on input image)

        If image is provided, Qwen encodes both image + text into prompt_embeds,
        so SD3.5 generates based on the combined visual-textual semantics.

        Args:
            prompt: Text prompt for generation
            doc_id: Document ID for file naming
            task: Task name for file naming
            image: Optional input image - will be encoded by Qwen into prompt_embeds

        Returns:
            Tuple of (prompt, list_of_image_paths)
        """
        if self.mode != "generation":
            raise RuntimeError(
                "generate_image requires mode='generation'. "
                "Current mode is 'understanding'."
            )

        # Get prompt embeddings (with image if provided)
        # Qwen encodes image + text together, so prompt_embeds contains both
        negative_embeds, positive_embeds = self._get_prompt_embeds(prompt, image)

        if image is not None:
            eval_logger.debug(
                "Using Qwen to encode image + text into prompt_embeds for SD3.5"
            )

        # Set seed
        generator = torch.Generator(device=self._device).manual_seed(self.seed)

        # Generate image using Text2Img pipeline
        # (image info is already in prompt_embeds, no need for Img2Img)
        with torch.no_grad():
            output = self._pipe(
                prompt_embeds=positive_embeds,
                negative_prompt_embeds=negative_embeds,
                pooled_prompt_embeds=self._pooled_prompt_embeds,
                negative_pooled_prompt_embeds=self._pooled_prompt_embeds,
                height=self.sd3_height,
                width=self.sd3_width,
                num_inference_steps=self.sd3_num_inference_steps,
                guidance_scale=self.sd3_guidance_scale,
                generator=generator,
            )
            generated_image = output.images[0]

        # Save image
        output_images = []
        safe_filename = f"{task}_{doc_id}.png"
        image_path = os.path.join(self.output_image_dir, safe_filename)
        generated_image.save(image_path)
        output_images.append(image_path)
        eval_logger.info(f"Saved generated image: {image_path}")

        return prompt, output_images

    def format_output(self, text: str, images: List[str]) -> str:
        """Format output as JSON string"""
        output_dict = {"text": text, "images": images}
        return json.dumps(output_dict, ensure_ascii=False)

    def generate_until(self, requests: List[Instance]) -> List[str]:
        """Main inference method"""
        res = []
        pbar = tqdm(
            total=len(requests), disable=(self.rank != 0), desc="UAE Generating"
        )

        def get_uuid(task, split, doc_id):
            return f"{task}___{split}___{doc_id}"

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        # Group requests
        re_ords = utils.Collator(
            [reg.args for reg in requests], _collate, grouping=True
        )
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)

        for chunk in chunks:
            (
                contexts,
                all_gen_kwargs,
                doc_to_visual,
                doc_id,
                task,
                split,
            ) = zip(*chunk)
            task = task[0]
            split = split[0]
            doc_id = doc_id[0]
            context = contexts[0]

            doc_uuid = get_uuid(task, split, doc_id)

            # Check cache
            if self.continual_mode and self.cache_mode == "resume":
                if doc_uuid in self.response_cache:
                    content = self.response_cache[doc_uuid]
                    if content:
                        res.append(content)
                        pbar.update(1)
                        continue

            # Get visuals
            visuals = [
                doc_to_visual[0](self.task_dict[task][split][doc_id])
            ]
            visuals = self.flatten(visuals)

            # Extract image
            image = None
            if visuals:
                image = self._extract_image(visuals[-1])

            if self.mode == "understanding":
                # Image understanding mode
                output_text = self.understand_image(context, image, str(doc_id))
                formatted_output = output_text
            else:
                # Image generation mode
                output_text, output_images = self.generate_image(
                    context, str(doc_id), task, image
                )
                formatted_output = self.format_output(output_text, output_images)

            res.append(formatted_output)

            # Update cache
            if self.continual_mode:
                self.response_cache[doc_uuid] = formatted_output
                with open(self.response_persistent_file, "w") as f:
                    json.dump(self.response_cache, f, ensure_ascii=False, indent=2)

            pbar.update(1)

        # Reorder results
        res = re_ords.get_original(res)
        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """Not supported for generation models"""
        raise NotImplementedError(
            "UAE is a generation model and does not support loglikelihood"
        )

    def generate_until_multi_round(self, requests) -> List[str]:
        """Multi-round dialogue not yet implemented"""
        raise NotImplementedError(
            "Multi-round dialogue not yet implemented for UAE"
        )
