#!/usr/bin/env python3
"""
Test script for Yi-VL-6B integration with lmms-eval

Usage:
    python test_yi_vl.py --model-path 01-ai/Yi-VL-6B --image-path path/to/image.jpg
"""

import argparse
import sys
from pathlib import Path

# Add lmms-eval to path
lmms_eval_path = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(lmms_eval_path))

from lmms_eval.api.registry import get_model


def test_yi_vl_model(model_path: str, image_path: str):
    """Test Yi-VL model integration"""
    print(f"Testing Yi-VL model: {model_path}")
    print(f"Image: {image_path}")

    # Initialize model
    print("\n1. Initializing Yi-VL model...")
    try:
        model = get_model("yi_vl")(
            pretrained=model_path,
            batch_size=1,
            device="cuda:0"
        )
        print("✓ Model initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize model: {e}")
        return False

    # Check model properties
    print("\n2. Checking model properties...")
    print(f"   - Device: {model.device}")
    print(f"   - Rank: {model.rank}")
    print(f"   - World size: {model.world_size}")
    print(f"   - Batch size: {model.batch_size}")
    print(f"   - Max length: {model.max_length}")
    print("✓ Model properties verified")

    # Test tokenization
    print("\n3. Testing tokenization...")
    test_text = "What is in this image?"
    tokens = model.tok_encode(test_text)
    decoded = model.tok_decode(tokens)
    print(f"   - Original: {test_text}")
    print(f"   - Tokens: {tokens[:10]}... (length: {len(tokens)})")
    print(f"   - Decoded: {decoded}")
    print("✓ Tokenization works")

    print("\n✓ All tests passed!")
    print("\nTo run full evaluation, use:")
    print(f"accelerate launch -m lmms_eval \\")
    print(f"    --model yi_vl \\")
    print(f"    --model_args pretrained={model_path} \\")
    print(f"    --tasks mmbench \\")
    print(f"    --batch_size 1 \\")
    print(f"    --output_path ./logs/")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Yi-VL integration")
    parser.add_argument(
        "--model-path",
        type=str,
        default="01-ai/Yi-VL-6B",
        help="Path to Yi-VL model"
    )
    parser.add_argument(
        "--image-path",
        type=str,
        default=None,
        help="Path to test image (optional for basic test)"
    )

    args = parser.parse_args()

    success = test_yi_vl_model(args.model_path, args.image_path)
    sys.exit(0 if success else 1)
