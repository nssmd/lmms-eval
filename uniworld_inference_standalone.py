"""
Standalone inference script for UniWorld.
Used by lmms-eval subprocess wrapper.

This script runs in a separate conda environment to avoid dependency conflicts.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image


def load_model(model_path: str):
    """Load UniWorld model in its native environment"""
    # Import UniWorld dependencies (in isolated environment)
    from transformers import AutoProcessor
    from diffusers import FluxPipeline
    
    # Add UniWorld to path
    uniworld_path = Path(model_path).parent
    if str(uniworld_path) not in sys.path:
        sys.path.insert(0, str(uniworld_path))
    
    from modeling.uniworld import UnivaQwen2p5VLForConditionalGeneration
    
    # Load model
    model = UnivaQwen2p5VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    # Load processor
    processor = AutoProcessor.from_pretrained(model_path)
    
    # Load FLUX (if needed for visual CoT)
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        transformer=model.denoise_tower.denoiser,
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_slicing()
    
    return model, processor, pipe


def run_inference(
    model,
    processor,
    pipe,
    prompt: str,
    image_paths: list,
    mode: str,
    output_dir: str,
    doc_id: str,
    task: str,
    gen_kwargs: dict = None,
):
    """Run inference"""
    # Default generation kwargs
    if gen_kwargs is None:
        gen_kwargs = {}
    
    # Extract generation parameters
    max_new_tokens = gen_kwargs.get("max_new_tokens", 2048)
    do_sample = gen_kwargs.get("do_sample", False)
    temperature = gen_kwargs.get("temperature", 0.0)
    
    # Load images
    images = []
    for img_path in image_paths:
        if os.path.exists(img_path):
            images.append(Image.open(img_path).convert("RGB"))
    
    # Prepare inputs
    if images:
        inputs = processor(
            text=[prompt],
            images=images,
            return_tensors="pt",
        ).to(model.device)
    else:
        inputs = processor(
            text=[prompt],
            return_tensors="pt",
        ).to(model.device)
    
    # Generate
    with torch.inference_mode():
        if mode == "visual_cot":
            # Visual CoT: generate images
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
            )
            # TODO: Handle image generation with FLUX
            output_text = processor.batch_decode(
                output_ids, skip_special_tokens=True
            )[0]
            output_images = []
        else:
            # Text only
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
            )
            output_text = processor.batch_decode(
                output_ids, skip_special_tokens=True
            )[0]
            output_images = []
    
    return output_text, output_images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input JSON file")
    parser.add_argument("--output", required=True, help="Output JSON file")
    parser.add_argument("--model_path", required=True, help="UniWorld model path")
    args = parser.parse_args()
    
    # Load input
    with open(args.input, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    
    # Load model
    print(f"Loading UniWorld from {args.model_path}...")
    model, processor, pipe = load_model(args.model_path)
    print("Model loaded successfully")
    
    # Run inference
    print("Running inference...")
    text, images = run_inference(
        model,
        processor,
        pipe,
        input_data["prompt"],
        input_data["images"],
        input_data["mode"],
        input_data["output_dir"],
        input_data["doc_id"],
        input_data["task"],
        input_data.get("gen_kwargs", {}),
    )
    
    # Save output
    output_data = {
        "text": text,
        "images": images,
        "doc_id": input_data["doc_id"],
        "task": input_data["task"],
    }
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"Output saved to {args.output}")


if __name__ == "__main__":
    main()
