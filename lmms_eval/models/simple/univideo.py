import json
import os
import sys
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import yaml
from loguru import logger as eval_logger
from PIL import Image as PILImage
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

# Add UniVideo repository to Python path
# Expected: lmms-eval/UniVideo/ directory at project root
wd = Path(__file__).parent.parent.parent.parent.resolve()
univideo_path = os.path.join(str(wd), "UniVideo")
if os.path.exists(univideo_path):
    sys.path.append(univideo_path)
    eval_logger.info(f"Added UniVideo path to sys.path: {univideo_path}")
else:
    eval_logger.warning(
        f"UniVideo repository not found at {univideo_path}. "
        f"Please clone it or create a symlink."
    )


@register_model("univideo")
class UniVideo(lmms):
    """
    UniVideo Multimodal Model
    Supports video/image understanding and generation

    Multi-GPU Support:
        - GPU 0 (cuda:0): MLLM encoder (Qwen2.5-VL-7B)
        - GPU 1 (cuda:1): VAE + Transformer (HunyuanVideo)

    Modes:
        - "understanding": Visual understanding (image/video + text -> text)
        - "generation": Image/video generation (text/image -> image/video)

    Example usage for understanding:
    python -m lmms_eval \
        --model univideo \
        --model_args pretrained=/path/to/UniVideo,mode=understanding \
        --tasks videomme \
        --batch_size 1 \
        --output_path ./logs/

    Example usage for generation:
    python -m lmms_eval \
        --model univideo \
        --model_args pretrained=/path/to/UniVideo,mode=generation,task_type=i2i_edit \
        --tasks geneval \
        --batch_size 1 \
        --output_path ./logs/
    """

    def __init__(
        self,
        pretrained: str,
        mode: str = "understanding",
        task_type: str = "understanding",
        config_path: Optional[str] = None,
        output_image_dir: Optional[str] = None,
        max_new_tokens: int = 1000,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.0,
        image_guidance_scale: float = 2.0,
        timestep_shift: float = 7.0,
        height: int = 480,
        width: int = 832,
        num_frames: int = 1,
        seed: int = 42,
        continual_mode: bool = True,
        response_persistent_folder: Optional[str] = None,
        mllm_device: str = "cuda:0",
        diffusion_device: str = "cuda:1",
        **kwargs,
    ) -> None:
        super().__init__()

        # Validate mode
        if mode not in ["understanding", "generation"]:
            raise ValueError(
                f"mode must be 'understanding' or 'generation', got '{mode}'"
            )

        self.mode = mode
        self.task_type = task_type
        self.max_new_tokens = max_new_tokens
        self.pretrained = pretrained
        self.continual_mode = continual_mode

        # Multi-GPU device configuration
        self.mllm_device = torch.device(mllm_device)
        self.diffusion_device = torch.device(diffusion_device)
        eval_logger.info(f"MLLM device: {self.mllm_device}")
        eval_logger.info(f"Diffusion device (VAE + Transformer): {self.diffusion_device}")

        # Generation hyperparameters
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.image_guidance_scale = image_guidance_scale
        self.timestep_shift = timestep_shift
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.seed = seed

        # Import UniVideo dependencies
        try:
            from diffusers.models.autoencoders.autoencoder_kl_hunyuan_video import (
                AutoencoderKLHunyuanVideo,
            )
            from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

            from mllm_encoder import MLLMInContext, MLLMInContextConfig
            from pipeline_univideo import UniVideoPipeline, UniVideoPipelineConfig
            from transformer_univideo_hunyuan_video import (
                HunyuanVideoTransformer3DModel,
                TwoLayerMLP,
            )
            from utils import load_model

            self.AutoencoderKLHunyuanVideo = AutoencoderKLHunyuanVideo
            self.FlowMatchEulerDiscreteScheduler = FlowMatchEulerDiscreteScheduler
            self.MLLMInContext = MLLMInContext
            self.MLLMInContextConfig = MLLMInContextConfig
            self.UniVideoPipeline = UniVideoPipeline
            self.UniVideoPipelineConfig = UniVideoPipelineConfig
            self.HunyuanVideoTransformer3DModel = HunyuanVideoTransformer3DModel
            self.TwoLayerMLP = TwoLayerMLP
            self.load_model_fn = load_model

        except Exception as e:
            raise ImportError(
                f"Failed to import UniVideo dependencies. "
                f"Please ensure:\n"
                f"  1. UniVideo repository is available at lmms-eval root: "
                f"{univideo_path}\n"
                f"  2. All dependencies are installed (diffusers, transformers, etc.)\n"
                f"Error: {e}"
            )

        # Setup output directory
        if response_persistent_folder is None:
            self.response_persistent_folder = "./logs/univideo_persistent_folder"
        else:
            self.response_persistent_folder = response_persistent_folder

        if output_image_dir is None:
            self.output_image_dir = os.path.join(
                self.response_persistent_folder, "univideo_generated_images"
            )
        else:
            self.output_image_dir = output_image_dir

        os.makedirs(self.output_image_dir, exist_ok=True)
        eval_logger.info(f"Image output directory: {self.output_image_dir}")

        # Setup response cache for continual mode
        self.response_cache = {}
        self.cache_mode = "start"

        if self.continual_mode:
            os.makedirs(self.response_persistent_folder, exist_ok=True)
            self.response_persistent_file = os.path.join(
                self.response_persistent_folder, "univideo_response.json"
            )

            if os.path.exists(self.response_persistent_file):
                with open(self.response_persistent_file, "r") as f:
                    self.response_cache = json.load(f)
                self.cache_mode = "resume"
                eval_logger.info(f"Loaded cache: {len(self.response_cache)} records")

        # Setup config path - try multiple locations
        if config_path is None:
            # Try to find config in multiple locations
            possible_config_paths = [
                # 1. From pretrained path
                os.path.join(
                    pretrained, "configs/univideo_qwen2p5vl7b_hidden_hunyuanvideo.yaml"
                ),
                # 2. From UniVideo repo in lmms-eval
                os.path.join(
                    univideo_path, "configs/univideo_qwen2p5vl7b_hidden_hunyuanvideo.yaml"
                ),
                # 3. Direct path if pretrained is the config file itself
                pretrained,
            ]

            self.config_path = None
            for path in possible_config_paths:
                if os.path.exists(path) and path.endswith(".yaml"):
                    self.config_path = path
                    eval_logger.info(f"Found config at: {path}")
                    break

            if self.config_path is None:
                raise FileNotFoundError(
                    f"Could not find UniVideo config file. Searched in:\n"
                    f"  1. {possible_config_paths[0]}\n"
                    f"  2. {possible_config_paths[1]}\n"
                    f"Please specify config_path explicitly or ensure UniVideo is properly set up."
                )
        else:
            self.config_path = config_path

        # Setup rank and world size
        self._rank = 0
        self._world_size = 1

        # Load model
        eval_logger.info(f"Loading UniVideo model from {pretrained}")
        self._load_model()

        eval_logger.info("UniVideo model initialized successfully")

    def _load_model(self):
        """
        Load UniVideo model components on multiple GPUs:
        - MLLM encoder on mllm_device (default: cuda:0)
        - VAE + Transformer on diffusion_device (default: cuda:1)
        """
        # Load config
        with open(self.config_path, "r") as f:
            raw = yaml.safe_load(f)

        if "mllm_config" not in raw:
            raise KeyError("Missing required config section: mllm_config")
        if "pipeline_config" not in raw:
            raise KeyError("Missing required config section: pipeline_config")

        mllm_config = self.MLLMInContextConfig(**raw["mllm_config"])
        pipe_cfg = self.UniVideoPipelineConfig(**raw["pipeline_config"])
        transformer_ckpt_path = raw.get("transformer_ckpt_path")
        mllm_encoder_ckpt_path = raw.get("mllm_encoder_ckpt", None)

        # Make checkpoint paths absolute - try multiple base paths
        def resolve_ckpt_path(ckpt_path):
            """Try to find checkpoint in multiple locations"""
            if ckpt_path is None:
                return None
            if os.path.isabs(ckpt_path) and os.path.exists(ckpt_path):
                return ckpt_path

            # Try multiple base paths
            base_paths = [
                self.pretrained,  # User-specified pretrained path
                univideo_path,    # UniVideo repo in lmms-eval
                os.path.dirname(self.config_path),  # Same dir as config
                os.path.dirname(os.path.dirname(self.config_path)),  # Parent of config dir
            ]

            for base in base_paths:
                full_path = os.path.join(base, ckpt_path)
                if os.path.exists(full_path):
                    eval_logger.info(f"Found checkpoint at: {full_path}")
                    return full_path

            # Return original path (may fail later, but with clear error)
            eval_logger.warning(f"Checkpoint not found: {ckpt_path}")
            return ckpt_path

        transformer_ckpt_path = resolve_ckpt_path(transformer_ckpt_path)
        mllm_encoder_ckpt_path = resolve_ckpt_path(mllm_encoder_ckpt_path)

        # ============================================================
        # Load MLLM encoder on GPU 0 (mllm_device)
        # ============================================================
        eval_logger.info(f"[INIT] Loading MLLM encoder on {self.mllm_device}...")
        mllm_encoder = self.MLLMInContext(mllm_config)

        # Load mllm_encoder checkpoint if provided
        if mllm_encoder_ckpt_path is not None:
            eval_logger.info(
                f"[INIT] loading mllm_encoder ckpt from {mllm_encoder_ckpt_path}"
            )
            mllm_encoder = self.load_model_fn(mllm_encoder, mllm_encoder_ckpt_path)

        mllm_encoder.requires_grad_(False)
        mllm_encoder.eval()
        mllm_encoder = mllm_encoder.to(device=self.mllm_device, dtype=torch.bfloat16)
        eval_logger.info(f"[INIT] MLLM encoder loaded on {self.mllm_device}")

        # ============================================================
        # Load VAE on GPU 1 (diffusion_device)
        # ============================================================
        eval_logger.info(f"[INIT] Loading VAE on {self.diffusion_device}...")
        vae = self.AutoencoderKLHunyuanVideo.from_pretrained(
            pipe_cfg.hunyuan_model_id,
            subfolder="vae",
            low_cpu_mem_usage=False,
            device_map=None,
        )
        vae.eval()
        vae = vae.to(device=self.diffusion_device, dtype=torch.bfloat16)
        eval_logger.info(f"[INIT] VAE loaded on {self.diffusion_device}")

        # ============================================================
        # Load Transformer on GPU 1 (diffusion_device)
        # ============================================================
        eval_logger.info(f"[INIT] Loading Transformer on {self.diffusion_device}...")
        qwenvl_txt_dim = 3584
        transformer = self.HunyuanVideoTransformer3DModel.from_pretrained(
            pipe_cfg.hunyuan_model_id,
            subfolder="transformer",
            low_cpu_mem_usage=False,
            device_map=None,
            text_embed_dim=qwenvl_txt_dim,
        )

        # Reinitialize qwen_project_in connector
        transformer.qwen_project_in = self.TwoLayerMLP(
            qwenvl_txt_dim, qwenvl_txt_dim * 4, 4096
        )
        with torch.no_grad():
            torch.nn.init.ones_(transformer.qwen_project_in.ln.weight)
            for layer in transformer.qwen_project_in.mlp:
                if isinstance(layer, torch.nn.Linear):
                    torch.nn.init.xavier_uniform_(layer.weight, gain=1.0)
                    if layer.bias is not None:
                        torch.nn.init.zeros_(layer.bias)
        eval_logger.info(
            f"[INIT] Reinitialized qwen_project_in ({qwenvl_txt_dim} -> {qwenvl_txt_dim * 4} -> 4096)"
        )

        # Load transformer checkpoint
        def rename_func(state_dict):
            new_state_dict = {}
            for k, v in state_dict.items():
                new_k = (
                    k.replace("transformer.", "", 1)
                    if k.startswith("transformer.")
                    else k
                )
                new_state_dict[new_k] = v
            return new_state_dict

        if isinstance(transformer_ckpt_path, str) and os.path.exists(
            transformer_ckpt_path
        ):
            eval_logger.info(f"[INIT] loading ckpt from {transformer_ckpt_path}")
            transformer = self.load_model_fn(
                transformer, transformer_ckpt_path, rename_func=rename_func
            )

        transformer.eval()
        transformer = transformer.to(device=self.diffusion_device, dtype=torch.bfloat16)
        eval_logger.info(f"[INIT] Transformer loaded on {self.diffusion_device}")

        # ============================================================
        # Load Scheduler (CPU, lightweight)
        # ============================================================
        scheduler = self.FlowMatchEulerDiscreteScheduler.from_pretrained(
            pipe_cfg.hunyuan_model_id, subfolder="scheduler"
        )

        # ============================================================
        # Build UniVideo pipeline with multi-GPU support
        # ============================================================
        eval_logger.info("[INIT] Building UniVideo pipeline with multi-GPU support...")

        # Store components separately for multi-GPU inference
        self.mllm_encoder = mllm_encoder
        self.vae = vae
        self.transformer = transformer
        self.scheduler = scheduler
        self.pipe_cfg = pipe_cfg

        # Build pipeline - note: we'll handle device placement manually
        self.pipeline = self.UniVideoPipeline(
            transformer=transformer,
            vae=vae,
            scheduler=scheduler,
            mllm_encoder=mllm_encoder,
            univideo_config=pipe_cfg,
        )

        # Override the pipeline's device handling
        # The pipeline will use the devices we've already set for each component
        # Note: _execution_device is a read-only property, so we need to patch it
        # to return the diffusion device for generation tasks

        # Store device references for patching
        mllm_device = self.mllm_device
        diffusion_device = self.diffusion_device

        # Patch _execution_device to return diffusion_device
        # This ensures latents and other tensors are created on the correct device
        def make_execution_device_property(device):
            @property
            def patched_execution_device(self):
                return device
            return patched_execution_device

        # Monkey-patch the property
        type(self.pipeline)._execution_device = make_execution_device_property(diffusion_device)

        # Patch the pipeline's get_mllm_prompt_embeddings method to handle multi-GPU
        original_get_mllm_prompt_embeddings = self.pipeline.get_mllm_prompt_embeddings

        @torch.no_grad()
        def patched_get_mllm_prompt_embeddings(prompts, images=None, videos=None, device=None, dtype=None):
            """
            Patched version that handles multi-GPU data transfer:
            1. Move inputs to MLLM device (GPU 0)
            2. Run MLLM encoding
            3. Move outputs to diffusion device (GPU 1)
            """
            if prompts is None:
                raise ValueError("prompts must be provided")

            # Use MLLM tokenizer
            tokenize_fn = self.pipeline.mllm_encoder.get_tokenize_fn()
            tokenizer = self.pipeline.mllm_encoder.get_tokenizer()

            if not images:
                images = None
            if not videos:
                videos = None

            batch = tokenize_fn(tokenizer, prompts, images, videos)

            # Move inputs to MLLM device (GPU 0)
            inputs = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(mllm_device)
                else:
                    inputs[k] = v

            # MLLM encoding on GPU 0
            prompt_embeds, prompt_attention_mask = self.pipeline.mllm_encoder.encode_condition(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                pixel_values=inputs.get("pixel_values"),
                image_grid_thw=inputs.get("image_grid_thw"),
                pixel_values_videos=inputs.get("pixel_values_videos"),
                video_grid_thw=inputs.get("video_grid_thw"),
                second_per_grid_ts=inputs.get("second_per_grid_ts"),
            )

            # Move outputs to diffusion device (GPU 1) for transformer
            prompt_embeds = prompt_embeds.to(device=diffusion_device, dtype=dtype)
            prompt_attention_mask = prompt_attention_mask.to(device=diffusion_device)

            return prompt_embeds, prompt_attention_mask

        # Replace the method
        self.pipeline.get_mllm_prompt_embeddings = patched_get_mllm_prompt_embeddings

        # Also patch mllm_generation for understanding tasks
        original_mllm_generation = self.pipeline.mllm_generation

        @torch.no_grad()
        def patched_mllm_generation(prompts, images=None, videos=None, device=None, dtype=None):
            """
            Patched version for understanding tasks that handles multi-GPU.
            """
            if prompts is None:
                raise ValueError("prompts must be provided")

            tokenize_fn = self.pipeline.mllm_encoder.get_tokenize_fn()
            tokenizer = self.pipeline.mllm_encoder.get_tokenizer()

            if not images:
                images = None
            if not videos:
                videos = None

            batch = tokenize_fn(tokenizer, prompts, images, videos, add_queires=False)

            # Move inputs to MLLM device (GPU 0)
            inputs = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(mllm_device)
                else:
                    inputs[k] = v

            # Run generation on MLLM device
            output_text = self.pipeline.mllm_encoder.generation(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                pixel_values=inputs.get("pixel_values"),
                image_grid_thw=inputs.get("image_grid_thw"),
                pixel_values_videos=inputs.get("pixel_values_videos"),
                video_grid_thw=inputs.get("video_grid_thw"),
                second_per_grid_ts=inputs.get("second_per_grid_ts"),
            )
            return output_text

        self.pipeline.mllm_generation = patched_mllm_generation

        self._model = self.pipeline
        self._tokenizer = mllm_encoder.get_tokenizer()

        # Log GPU memory usage
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                eval_logger.info(
                    f"[GPU {i}] Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
                )

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    @property
    def model(self):
        return self._model

    @property
    def tokenizer(self):
        return self._tokenizer

    def set_seed(self, seed: int):
        """Set random seeds for reproducibility"""
        if seed > 0:
            import random

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def understand_visual(
        self, prompt: str, image=None, video=None, doc_id: str = ""
    ) -> str:
        """
        Understand image/video and answer question

        Args:
            prompt: Input text prompt/question
            image: PIL Image to understand (optional)
            video: Video path or tensor to understand (optional)
            doc_id: Document ID for logging

        Returns:
            Generated text answer
        """
        self.set_seed(self.seed)

        # Prepare pipeline kwargs
        pipeline_kwargs = {
            "prompts": [prompt],
            "seed": self.seed,
            "task": "understanding",
        }

        # Add image or video condition
        if image is not None:
            # Save image temporarily for pipeline
            # Convert to RGB to handle RGBA images (JPEG doesn't support alpha channel)
            import tempfile

            rgb_image = image.convert("RGB") if hasattr(image, "convert") else image
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                rgb_image.save(tmp.name)
                pipeline_kwargs["cond_image_path"] = tmp.name

        if video is not None:
            if isinstance(video, str):
                pipeline_kwargs["cond_video_path"] = video
            else:
                # Save video temporarily
                import tempfile

                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                    # Assume video is a path or needs to be saved
                    pipeline_kwargs["cond_video_path"] = video

        # Run pipeline
        output = self.pipeline(**pipeline_kwargs)

        # Extract text output
        if hasattr(output, "text") and output.text is not None:
            return output.text[0] if output.text else ""
        return ""

    def generate_image(
        self, prompt: str, cond_image=None, doc_id: str = "", task: str = ""
    ) -> Tuple[str, List[str]]:
        """
        Generate image from prompt (optionally conditioned on input image)

        Args:
            prompt: Input text prompt
            cond_image: Optional conditioning image for editing
            doc_id: Document ID for file naming
            task: Task name for file naming

        Returns:
            Tuple of (generated_text, list_of_image_paths)
        """
        self.set_seed(self.seed)

        # Negative prompt for generation
        negative_prompt = (
            "Bright tones, overexposed, oversharpening, static, blurred details, "
            "subtitles, style, works, paintings, images, static, overall gray, "
            "worst quality, low quality, JPEG compression residue, ugly, incomplete, "
            "extra fingers, poorly drawn hands, poorly drawn faces, deformed, "
            "disfigured, misshapen limbs, fused fingers, still picture, messy background"
        )

        # Prepare pipeline kwargs
        pipeline_kwargs = {
            "prompts": [prompt],
            "negative_prompt": negative_prompt,
            "height": self.height,
            "width": self.width,
            "num_frames": self.num_frames,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "image_guidance_scale": self.image_guidance_scale,
            "seed": self.seed,
            "timestep_shift": self.timestep_shift,
            "task": self.task_type,
        }

        # Add conditioning image if provided
        if cond_image is not None:
            import tempfile

            # Convert to RGB to handle RGBA images (JPEG doesn't support alpha channel)
            rgb_image = cond_image.convert("RGB") if hasattr(cond_image, "convert") else cond_image
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                rgb_image.save(tmp.name)
                pipeline_kwargs["cond_image_path"] = tmp.name

        # Run pipeline
        output = self.pipeline(**pipeline_kwargs)

        # Save output image
        output_images = []
        if hasattr(output, "frames") and output.frames is not None:
            frames = output.frames[0]  # (F, H, W, C)
            if hasattr(frames, "detach"):
                frames = frames.detach().cpu().float().numpy()

            from PIL import Image as PILImage

            F, H, W, C = frames.shape
            if F == 1:
                img = frames[0]
                if img.min() < 0:
                    img = (img + 1.0) / 2.0
                img = (img * 255).clip(0, 255).astype(np.uint8)
                safe_filename = f"{task}_{doc_id}.png"
                image_path = os.path.join(self.output_image_dir, safe_filename)
                PILImage.fromarray(img).save(image_path)
                output_images.append(image_path)
                eval_logger.info(f"Saved image: {image_path}")

        return "", output_images

    def format_output(self, text: str, images: List[str]) -> str:
        """Format output as JSON string"""
        output_dict = {"text": text, "images": images}
        return json.dumps(output_dict, ensure_ascii=False)

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
        """Main inference method"""
        res = []
        pbar = tqdm(
            total=len(requests), disable=(self.rank != 0), desc="UniVideo Generating"
        )

        def get_uuid(task, split, doc_id):
            return f"{task}___{split}___{doc_id}"

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [
            reg.args for reg in requests
        ]:
            doc_uuid = get_uuid(task, split, doc_id)

            # Check cache
            if self.continual_mode and self.cache_mode == "resume":
                if doc_uuid in self.response_cache:
                    content = self.response_cache[doc_uuid]
                    if content:
                        res.append(content)
                        pbar.update(1)
                        continue

            prompt = contexts

            # Check if this is Uni-MMMU interleaved generation mode
            bagel_interleaved = gen_kwargs.get("bagel_interleaved", None)

            if bagel_interleaved is not None:
                # Uni-MMMU interleaved generation mode
                # Get input images and doc data
                doc = self.task_dict[task][split][doc_id]
                input_images = []
                if doc_to_visual is not None:
                    visuals = [doc_to_visual(doc)]
                    input_images = self.flatten(visuals)

                output_text, output_images = self.generate_uni_mmmu_interleaved(
                    input_images, prompt, str(doc_id), task, bagel_interleaved, doc
                )
                formatted_output = self.format_output(output_text, output_images)

            elif self.mode == "understanding":
                # Visual understanding mode
                image = None
                video = None

                if doc_to_visual is not None:
                    visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
                    visuals = self.flatten(visuals)

                    if visuals and len(visuals) > 0:
                        visual = visuals[0]
                        # Check if it's a video path or image
                        if isinstance(visual, str) and visual.endswith(
                            (".mp4", ".avi", ".mov", ".mkv")
                        ):
                            video = visual
                        else:
                            image = visual

                output_text = self.understand_visual(
                    prompt, image=image, video=video, doc_id=str(doc_id)
                )
                formatted_output = output_text

            else:
                # Generation mode
                cond_image = None
                if doc_to_visual is not None:
                    visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
                    visuals = self.flatten(visuals)
                    if visuals and len(visuals) > 0:
                        cond_image = visuals[0]

                output_text, output_images = self.generate_image(
                    prompt, cond_image=cond_image, doc_id=str(doc_id), task=task
                )
                formatted_output = self.format_output(output_text, output_images)

            res.append(formatted_output)

            # Update cache
            if self.continual_mode:
                self.response_cache[doc_uuid] = formatted_output
                with open(self.response_persistent_file, "w") as f:
                    json.dump(self.response_cache, f, ensure_ascii=False, indent=2)

            pbar.update(1)

        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """Not supported for generation models"""
        raise NotImplementedError(
            "UniVideo is a generation model and does not support loglikelihood"
        )

    def generate_until_multi_round(self, requests) -> List[str]:
        """Multi-round dialogue generation"""
        raise NotImplementedError("TODO: Implement multi-round dialogue generation")

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
        Uni-MMMU interleaved generation aligned with original benchmark.

        This implements the generation flow for Uni-MMMU:
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
        self.set_seed(self.seed)

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
        num_inference_steps = interleaved_config.get("num_inference_steps", self.num_inference_steps)
        guidance_scale = interleaved_config.get("guidance_scale", self.guidance_scale)
        image_guidance_scale = interleaved_config.get("image_guidance_scale", self.image_guidance_scale)

        generated_images = []

        # Negative prompt for generation
        negative_prompt = (
            "Bright tones, overexposed, oversharpening, static, blurred details, "
            "subtitles, style, works, paintings, images, static, overall gray, "
            "worst quality, low quality, JPEG compression residue, ugly, incomplete, "
            "extra fingers, poorly drawn hands, poorly drawn faces, deformed, "
            "disfigured, misshapen limbs, fused fingers, still picture, messy background"
        )

        if task_type == "jigsaw":
            # Jigsaw: Generate 2 completed images then final answer
            # Image 1: Candidate 0 completion
            suffix1 = "Output ONLY a single image with Candidate 0 placed in the bottom-right cell. No text."
            gen_prompt1 = prompt + "\n" + suffix1

            # Prepare input images for MLLM
            mllm_images = [[img for img in input_images if img is not None]] if input_images else None

            # Generate image for candidate 0
            img0_path = self._generate_single_image(
                prompt=gen_prompt1,
                input_images=mllm_images,
                doc_id=doc_id,
                task=task,
                suffix="_cand0",
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                image_guidance_scale=image_guidance_scale,
                negative_prompt=negative_prompt,
            )
            if img0_path:
                generated_images.append(img0_path)

            # Image 2: Candidate 1 completion
            suffix2 = "Output ONLY a single image with Candidate 1 placed in the bottom-right cell. No text."
            gen_prompt2 = prompt + "\n" + suffix2

            img1_path = self._generate_single_image(
                prompt=gen_prompt2,
                input_images=mllm_images,
                doc_id=doc_id,
                task=task,
                suffix="_cand1",
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                image_guidance_scale=image_guidance_scale,
                negative_prompt=negative_prompt,
            )
            if img1_path:
                generated_images.append(img1_path)

            # Final answer: Compare the two generated images
            # Build context with original images + generated images
            final_images = []
            for img in input_images:
                if img is not None:
                    final_images.append(img)

            # Load generated images
            if img0_path and os.path.exists(img0_path):
                final_images.append(PILImage.open(img0_path).convert("RGB"))
            if img1_path and os.path.exists(img1_path):
                final_images.append(PILImage.open(img1_path).convert("RGB"))

            final_suffix = (
                'Now output EXACTLY ONE <FINAL_ANSWER_JSON>{"choice": 0 or 1, "rationale": "≤30 words"}</FINAL_ANSWER_JSON>\n'
                "Do not output any additional images."
            )
            final_prompt = prompt + "\nCOMPLETED WITH CANDIDATE 0:\nCOMPLETED WITH CANDIDATE 1:\n" + final_suffix

            final_text = self._generate_text_with_images(
                prompt=final_prompt,
                images=final_images,
            )

        else:
            # Maze/Sliding: [gen_text(plan) → gen_image(step)]×k → gen_text(answer)
            step_texts = []
            step_images = []
            current_images = [img for img in input_images if img is not None]

            for i in range(1, num_images + 1):
                # Generate planning text
                if task_type == "maze":
                    plan_suffix = f'Now planning for step {i}, Please output a sentence in the form: "Next, move one step up/down/left/right."'
                else:  # sliding
                    plan_suffix = f'Now planning for step {i}, Please output a sentence describing which tile to move and in which direction.'

                plan_prompt = prompt
                for j, txt in enumerate(step_texts):
                    plan_prompt += f"\nStep {j+1} plan: {txt}"
                plan_prompt += "\n" + plan_suffix

                plan_text = self._generate_text_with_images(
                    prompt=plan_prompt,
                    images=current_images,
                )
                eval_logger.info(f"Step {i} plan: {plan_text}")
                step_texts.append(plan_text)

                # Generate step image
                img_suffix = f"Now, generate the image for step {i}."
                img_prompt = prompt
                for j, txt in enumerate(step_texts):
                    img_prompt += f"\nStep {j+1} plan: {txt}"
                img_prompt += "\n" + img_suffix

                mllm_images = [current_images] if current_images else None

                img_path = self._generate_single_image(
                    prompt=img_prompt,
                    input_images=mllm_images,
                    doc_id=doc_id,
                    task=task,
                    suffix=f"_step_{i:04d}",
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    image_guidance_scale=image_guidance_scale,
                    negative_prompt=negative_prompt,
                )
                if img_path:
                    generated_images.append(img_path)
                    step_images.append(img_path)
                    # Add generated image to context for next iteration
                    if os.path.exists(img_path):
                        current_images.append(PILImage.open(img_path).convert("RGB"))
                    eval_logger.info(f"Saved step {i} image: {img_path}")

            # Final answer
            final_images = [img for img in input_images if img is not None]
            for img_path in step_images:
                if os.path.exists(img_path):
                    final_images.append(PILImage.open(img_path).convert("RGB"))

            final_suffix = (
                "After the images, emit EXACTLY ONE LINE containing ONLY the final move list "
                "as <ANSWER_JSON>[...]</ANSWER_JSON>. No other text."
            )
            final_prompt = prompt
            for j, txt in enumerate(step_texts):
                final_prompt += f"\nStep {j+1} plan: {txt}"
                final_prompt += f"\nImage for step {j+1}:"
            final_prompt += "\n" + final_suffix

            final_text = self._generate_text_with_images(
                prompt=final_prompt,
                images=final_images,
            )
            eval_logger.info(f"Maze/Sliding final answer: {final_text}")

        return final_text, generated_images

    def _generate_text_with_images(
        self,
        prompt: str,
        images: List = None,
    ) -> str:
        """
        Generate text using MLLM with optional image context.

        Args:
            prompt: Text prompt
            images: List of PIL images

        Returns:
            Generated text
        """
        # Prepare images for MLLM
        mllm_images = None
        if images and len(images) > 0:
            mllm_images = [[img for img in images if img is not None]]
            if not mllm_images[0]:
                mllm_images = None

        # Use pipeline's mllm_generation method
        try:
            output_text = self.pipeline.mllm_generation(
                prompts=[prompt],
                images=mllm_images,
                videos=None,
                device=self.mllm_device,
                dtype=torch.bfloat16,
            )
            if output_text and len(output_text) > 0:
                return output_text[0]
            return ""
        except Exception as e:
            eval_logger.error(f"Text generation failed: {e}")
            return ""

    def _generate_single_image(
        self,
        prompt: str,
        input_images: List = None,
        doc_id: str = "",
        task: str = "",
        suffix: str = "",
        num_inference_steps: int = 50,
        guidance_scale: float = 7.0,
        image_guidance_scale: float = 2.0,
        negative_prompt: str = "",
    ) -> Optional[str]:
        """
        Generate a single image using the pipeline.

        Args:
            prompt: Text prompt for generation
            input_images: Optional list of reference images [[PIL.Image, ...]]
            doc_id: Document ID for file naming
            task: Task name for file naming
            suffix: Suffix for filename
            num_inference_steps: Number of denoising steps
            guidance_scale: Text guidance scale
            image_guidance_scale: Image guidance scale
            negative_prompt: Negative prompt

        Returns:
            Path to saved image or None
        """
        try:
            # Prepare pipeline kwargs
            pipeline_kwargs = {
                "prompts": [prompt],
                "negative_prompt": negative_prompt,
                "height": self.height,
                "width": self.width,
                "num_frames": 1,  # Single image
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "image_guidance_scale": image_guidance_scale,
                "seed": self.seed,
                "timestep_shift": self.timestep_shift,
                "task": "t2i",  # Text to image
            }

            # Add reference images if provided
            if input_images and len(input_images) > 0 and len(input_images[0]) > 0:
                pipeline_kwargs["ref_images"] = input_images
                pipeline_kwargs["task"] = "multiid"  # Use multiid task for reference images

            # Run pipeline
            output = self.pipeline(**pipeline_kwargs)

            # Save output image
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
                    safe_filename = f"{task}_{doc_id}{suffix}.png"
                    image_path = os.path.join(self.output_image_dir, safe_filename)
                    PILImage.fromarray(img).save(image_path)
                    eval_logger.info(f"Saved image: {image_path}")
                    return image_path

            return None
        except Exception as e:
            eval_logger.error(f"Image generation failed: {e}")
            return None
