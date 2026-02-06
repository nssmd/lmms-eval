#!/usr/bin/env python3
"""
Visual CoT ROVER 评测示例脚本

用法:
    # 方式1: 从图像目录读取原始图像
    python examples/rover_eval/evaluate_visual_cot.py \
        --log_dir ./logs/bagel_visual_cot/ \
        --image_dir ./dataset/illusionbench/images/ \
        --output visual_cot_results.csv

    # 方式2: 从 parquet 文件读取原始图像 (适用于 PhyX 等数据集)
    python examples/rover_eval/evaluate_visual_cot.py \
        --log_dir ./logs/bagel_visual_cot/phyx_mechanics100_visual_cot/ \
        --parquet_file /path/to/phyx_mechanics100.parquet \
        --base_dir /home/aiscuser/data/zwb \
        --output visual_cot_results.csv
"""

import argparse
import base64
import io
import json
import sys
from pathlib import Path

from PIL import Image

from lmms_eval.rover_eval import VisualCoTEvaluator


def decode_base64_to_image(base64_string: str) -> Image.Image:
    """Decode base64 string to PIL Image."""
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    return image


def load_parquet_images(parquet_path: str) -> dict:
    """Load images from parquet file, indexed by row index (doc_id)."""
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    images = {}
    for idx, row in df.iterrows():
        if "image" in row:
            try:
                images[idx] = decode_base64_to_image(row["image"])
            except Exception as e:
                print(f"Warning: Failed to decode image for index {idx}: {e}")
    return images


def main():
    parser = argparse.ArgumentParser(description="Evaluate Visual CoT outputs using ROVER")
    parser.add_argument("--log_dir", type=str, required=True, help="Directory containing JSON metadata files")
    parser.add_argument("--image_dir", type=str, default=None, help="Directory containing original question images")
    parser.add_argument("--parquet_file", type=str, default=None, help="Parquet file containing original images (base64 encoded)")
    parser.add_argument("--base_dir", type=str, default=None, help="Base directory for resolving relative paths in generated_images (e.g., ./logs/...)")
    parser.add_argument("--output", type=str, default="visual_cot_results.csv", help="Output CSV file")
    parser.add_argument("--metrics", nargs="+", default=["ra", "al"], choices=["ra", "al"], help="Metrics to evaluate")
    parser.add_argument("--task_category", type=str, default=None,
                       choices=["real_world", "mathematical", "stem", "puzzles", "chart_table", "spatial", "perception"],
                       help="Task category for customized prompts")
    parser.add_argument("--max_workers", type=int, default=10, help="Max parallel workers")
    parser.add_argument("--max_retries", type=int, default=3, help="Max retries per evaluation")

    args = parser.parse_args()

    # Validate arguments
    if args.image_dir is None and args.parquet_file is None:
        print("Error: Must specify either --image_dir or --parquet_file")
        sys.exit(1)

    log_dir = Path(args.log_dir)

    if not log_dir.exists():
        print(f"Error: Log directory {log_dir} not found")
        sys.exit(1)

    # Load parquet images if specified
    parquet_images = None
    if args.parquet_file:
        parquet_path = Path(args.parquet_file)
        if not parquet_path.exists():
            print(f"Error: Parquet file {parquet_path} not found")
            sys.exit(1)
        print(f"Loading images from parquet: {parquet_path}")
        parquet_images = load_parquet_images(str(parquet_path))
        print(f"Loaded {len(parquet_images)} images from parquet")

    image_dir = Path(args.image_dir) if args.image_dir else None
    if image_dir and not image_dir.exists():
        print(f"Error: Image directory {image_dir} not found")
        sys.exit(1)

    # 收集所有 JSON 文件
    json_files = sorted(log_dir.glob("*_metadata.json"))
    print(f"Found {len(json_files)} JSON files in {log_dir}")

    if len(json_files) == 0:
        print("No JSON files found. Exiting.")
        sys.exit(1)

    # 准备评测数据
    json_paths = []
    original_images = []

    for json_file in json_files:
        with open(json_file) as f:
            try:
                data = json.load(f)
                doc_id = data.get("doc_id", "unknown")

                original_img = None

                # 优先从 parquet 获取图像
                if parquet_images is not None:
                    # doc_id 可能是 int 或 str
                    doc_id_int = int(doc_id) if isinstance(doc_id, str) and str(doc_id).isdigit() else doc_id
                    if doc_id_int in parquet_images:
                        original_img = parquet_images[doc_id_int]
                    elif doc_id in parquet_images:
                        original_img = parquet_images[doc_id]

                # 否则从图像目录获取
                if original_img is None and image_dir is not None:
                    for ext in [".png", ".jpg", ".jpeg", ".PNG", ".JPG"]:
                        candidate = image_dir / f"{doc_id}{ext}"
                        if candidate.exists():
                            original_img = str(candidate)
                            break

                if original_img is None:
                    print(f"Warning: Image for doc_id {doc_id} not found, skipping")
                    continue

                json_paths.append(str(json_file))
                original_images.append(original_img)

            except Exception as e:
                print(f"Error loading {json_file}: {e}, skipping")
                continue

    if len(json_paths) == 0:
        print("No valid samples to evaluate. Exiting.")
        sys.exit(1)

    print(f"Evaluating {len(json_paths)} samples")
    print(f"  Metrics: {args.metrics}")
    if args.task_category:
        print(f"  Task Category: {args.task_category}")
    if args.base_dir:
        print(f"  Base Dir: {args.base_dir}")

    # 初始化评测器
    evaluator = VisualCoTEvaluator(
        metrics=args.metrics,
        task_category=args.task_category,
        max_retries=args.max_retries,
        base_dir=args.base_dir,
    )
    
    # 批量评测
    results = evaluator.evaluate_batch(
        json_paths=json_paths,
        original_images=original_images,
        max_workers=args.max_workers
    )
    
    # 保存结果
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    print(f"\n✅ Results saved to {args.output}")
    
    # 统计信息
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Total samples: {len(df)}")
    
    if "ra" in args.metrics:
        valid_ra = df['ra_score'].notna()
        print(f"\nRA Evaluation:")
        print(f"  Valid: {valid_ra.sum()}/{len(df)}")
        print(f"  Average Score: {df.loc[valid_ra, 'ra_score'].mean():.2f}")
        print(f"  Score Distribution:")
        for score in sorted(df.loc[valid_ra, 'ra_score'].unique()):
            count = (df['ra_score'] == score).sum()
            print(f"    {int(score)}: {count} samples ({count/valid_ra.sum()*100:.1f}%)")
    
    if "al" in args.metrics:
        valid_al = df['al_score'].notna()
        print(f"\nAL Evaluation:")
        print(f"  Valid: {valid_al.sum()}/{len(df)}")
        print(f"  Average Score: {df.loc[valid_al, 'al_score'].mean():.2f}")
        print(f"  Score Distribution:")
        for score in sorted(df.loc[valid_al, 'al_score'].unique()):
            count = (df['al_score'] == score).sum()
            print(f"    {int(score)}: {count} samples ({count/valid_al.sum()*100:.1f}%)")
    
    print("="*60)


if __name__ == "__main__":
    main()
