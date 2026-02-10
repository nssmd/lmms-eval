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

    # 方式3: 从 HuggingFace 加载数据集
    python examples/rover_eval/evaluate_visual_cot.py \
        --log_dir ./logs/bagel_visual_cot/geometry3k_visual_cot/ \
        --hf_dataset zwb20060413/geometry3k \
        --hf_split test \
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


def load_parquet_images(parquet_path: str, multi_image_columns: list = None) -> dict:
    """
    Load images from parquet file, indexed by row index (doc_id).
    
    Args:
        parquet_path: Path to parquet file
        multi_image_columns: List of image column names for multi-image tasks
    
    Returns:
        Dict mapping doc_id to image or list of images
    """
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    images = {}

    # Handle multi-image columns (e.g., uni_mmmu jigsaw)
    if multi_image_columns:
        print(f"Loading multi-image columns: {multi_image_columns}")
        for idx, row in df.iterrows():
            img_list = []
            for col_name in multi_image_columns:
                if col_name not in df.columns:
                    print(f"Warning: Column '{col_name}' not found in parquet")
                    continue
                try:
                    img = _extract_parquet_image(row[col_name], idx, col_name)
                    if img is not None:
                        img_list.append(img)
                except Exception as e:
                    print(f"Warning: Failed to decode image for index {idx}, column {col_name}: {e}")
            if img_list:
                images[idx] = img_list
        return images

    # Handle single image column
    # Try multiple possible image column names
    image_col = None
    for col_name in ["image", "images", "original_image", "test_image", "img", "picture", "initial_image"]:
        if col_name in df.columns:
            image_col = col_name
            break

    if image_col is None:
        print(f"Warning: No image column found in parquet. Columns: {df.columns.tolist()}")
        return images

    for idx, row in df.iterrows():
        try:
            img = _extract_parquet_image(row[image_col], idx, image_col)
            if img is not None:
                images[idx] = img
        except Exception as e:
            print(f"Warning: Failed to decode image for index {idx}: {e}")
    
    return images


def _extract_parquet_image(img_data, idx, col_name):
    """Extract a single PIL Image from parquet data."""
    if img_data is None:
        return None

    # Handle list/array of images (take first image)
    if isinstance(img_data, (list, tuple)) or (hasattr(img_data, '__len__') and hasattr(img_data, '__getitem__') and not isinstance(img_data, (str, bytes, dict))):
        if len(img_data) > 0:
            img_data = img_data[0]
        else:
            return None

    if isinstance(img_data, str):
        # base64 encoded
        return decode_base64_to_image(img_data)
    elif isinstance(img_data, dict):
        # HF image format - check if bytes is None (empty parquet)
        if "bytes" in img_data and img_data["bytes"] is not None:
            return Image.open(io.BytesIO(img_data["bytes"])).convert("RGB")
        elif "path" in img_data and img_data["path"] is None and "bytes" in img_data and img_data["bytes"] is None:
            # Empty HF image format - skip this entry
            return None
    elif isinstance(img_data, bytes):
        return Image.open(io.BytesIO(img_data)).convert("RGB")
    elif isinstance(img_data, Image.Image):
        return img_data.convert("RGB")
    
    return None


def load_hf_dataset_images(dataset_name: str, split: str = "test", image_column: str = "image", config_name: str = None, filter_column: str = None, filter_value: str = None, multi_image_columns: list = None) -> dict:
    """
    Load images from HuggingFace dataset, indexed by row index (doc_id).
    
    Args:
        dataset_name: HF dataset name
        split: Dataset split
        image_column: Primary image column name
        config_name: Dataset config name
        filter_column: Column to filter by
        filter_value: Value to filter
        multi_image_columns: List of image column names for multi-image tasks (e.g., ["ref_image", "cand0_image", "cand1_image"])
    
    Returns:
        Dict mapping doc_id to image or list of images
    """
    from datasets import load_dataset

    print(f"Loading HuggingFace dataset: {dataset_name}, config: {config_name}, split: {split}")
    dataset = load_dataset(dataset_name, config_name, split=split)

    # Apply filter if specified
    if filter_column and filter_value:
        print(f"Filtering: {filter_column} == {filter_value}")
        dataset = dataset.filter(lambda x: x[filter_column] == filter_value)
        print(f"After filtering: {len(dataset)} samples")

    images = {}
    for idx, row in enumerate(dataset):
        try:
            # Handle multi-image columns (e.g., uni_mmmu jigsaw)
            if multi_image_columns:
                img_list = []
                for col in multi_image_columns:
                    img_data = row.get(col)
                    if img_data is not None:
                        img = _extract_single_image(img_data, idx, col)
                        if img is not None:
                            img_list.append(img)
                if img_list:
                    images[idx] = img_list
                continue
            
            # Handle single image column
            img = row.get(image_column)
            if img is None:
                print(f"Warning: No image column '{image_column}' for index {idx}")
                continue

            extracted_img = _extract_single_image(img, idx, image_column)
            if extracted_img is not None:
                images[idx] = extracted_img
                
        except Exception as e:
            print(f"Warning: Failed to load image for index {idx}: {e}")

    print(f"Loaded {len(images)} image entries from HuggingFace dataset")
    return images


def _extract_single_image(img_data, idx, column_name):
    """Extract a single PIL Image from various formats."""
    try:
        if isinstance(img_data, Image.Image):
            return img_data.convert("RGB")
        elif isinstance(img_data, str):
            # base64 encoded
            return decode_base64_to_image(img_data)
        elif isinstance(img_data, dict) and "bytes" in img_data and img_data["bytes"] is not None:
            # HF image format with bytes
            return Image.open(io.BytesIO(img_data["bytes"])).convert("RGB")
        elif isinstance(img_data, bytes):
            return Image.open(io.BytesIO(img_data)).convert("RGB")
        else:
            print(f"Warning: Unknown image format for index {idx}, column {column_name}: {type(img_data)}")
            return None
    except Exception as e:
        print(f"Warning: Failed to extract image for index {idx}, column {column_name}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate Visual CoT outputs using ROVER")
    parser.add_argument("--log_dir", type=str, required=True, help="Directory containing JSON metadata files")
    parser.add_argument("--image_dir", type=str, default=None, help="Directory containing original question images")
    parser.add_argument("--parquet_file", type=str, default=None, help="Parquet file containing original images (base64 encoded)")
    parser.add_argument("--hf_dataset", type=str, default=None, help="HuggingFace dataset name (e.g., shasha/AuxSolidMath)")
    parser.add_argument("--hf_config", type=str, default=None, help="HuggingFace dataset config name (e.g., test_easy)")
    parser.add_argument("--hf_split", type=str, default="test", help="HuggingFace dataset split (default: test)")
    parser.add_argument("--hf_image_column", type=str, default="image", help="Image column name in HF dataset (default: image)")
    parser.add_argument("--hf_multi_image_columns", nargs="+", default=None, help="Multiple image column names for multi-image tasks (e.g., ref_image cand0_image cand1_image)")
    parser.add_argument("--hf_filter_column", type=str, default=None, help="Column name to filter HF dataset (e.g., type)")
    parser.add_argument("--hf_filter_value", type=str, default=None, help="Value to filter by (e.g., 'Fine-grained Discrimination')")
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
    if args.image_dir is None and args.parquet_file is None and args.hf_dataset is None:
        print("Error: Must specify one of --image_dir, --parquet_file, or --hf_dataset")
        sys.exit(1)

    log_dir = Path(args.log_dir)

    if not log_dir.exists():
        print(f"Error: Log directory {log_dir} not found")
        sys.exit(1)

    # Load images from different sources
    loaded_images = None

    # Priority 1: HuggingFace dataset
    if args.hf_dataset:
        loaded_images = load_hf_dataset_images(
            args.hf_dataset,
            split=args.hf_split,
            image_column=args.hf_image_column,
            config_name=args.hf_config,
            filter_column=args.hf_filter_column,
            filter_value=args.hf_filter_value,
            multi_image_columns=args.hf_multi_image_columns
        )

    # Priority 2: Parquet file
    if loaded_images is None and args.parquet_file:
        parquet_path = Path(args.parquet_file)
        if not parquet_path.exists():
            print(f"Error: Parquet file {parquet_path} not found")
            sys.exit(1)
        print(f"Loading images from parquet: {parquet_path}")
        
        # Detect multi-image columns from filename
        parquet_name = parquet_path.stem.lower()
        multi_cols = None
        if "jigsaw" in parquet_name:
            multi_cols = ["ref_image", "cand0_image", "cand1_image"]
        
        loaded_images = load_parquet_images(str(parquet_path), multi_image_columns=multi_cols)
        print(f"Loaded {len(loaded_images)} images from parquet")
        
        # If parquet images are empty, try to infer HF dataset from filename
        if len(loaded_images) == 0:
            print("Warning: Parquet file contains no valid images (empty bytes/path)")
            
            # Map parquet filename to HF dataset
            # Format: (dataset_name, config_name, split, image_column, multi_image_columns)
            hf_dataset_map = {
                "auxsolidmath_easy": ("shasha/AuxSolidMath", "test_easy", "test_easy", "original_image", None),
                "auxsolidmath_hard": ("shasha/AuxSolidMath", "test_hard", "test_hard", "original_image", None),
                "uni_mmmu_jigsaw100": ("zwb20060413/uni_mmmu", "jigsaw", "train", None, ["ref_image", "cand0_image", "cand1_image"]),
                "uni_mmmu_maze100": ("zwb20060413/uni_mmmu", "maze", "train", "initial_image", None),
                "uni_mmmu_sliding54": ("zwb20060413/uni_mmmu", "sliding", "train", "initial_image", None),
            }
            
            if parquet_name in hf_dataset_map:
                dataset_name, config_name, split, image_col, multi_cols = hf_dataset_map[parquet_name]
                print(f"Attempting to load from HuggingFace: {dataset_name} (config: {config_name}, split: {split})")
                try:
                    loaded_images = load_hf_dataset_images(
                        dataset_name,
                        split=split,
                        image_column=image_col if image_col else "image",
                        config_name=config_name,
                        multi_image_columns=multi_cols
                    )
                except Exception as e:
                    print(f"Failed to load from HuggingFace: {e}")
            else:
                print(f"No HuggingFace fallback available for: {parquet_name}")
                print("Please use --hf_dataset parameter to specify the dataset manually")

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

                # 优先从已加载的图像中获取 (HF 或 parquet)
                if loaded_images is not None:
                    # doc_id 可能是 int 或 str
                    doc_id_int = int(doc_id) if isinstance(doc_id, str) and str(doc_id).isdigit() else doc_id
                    if doc_id_int in loaded_images:
                        original_img = loaded_images[doc_id_int]
                    elif doc_id in loaded_images:
                        original_img = loaded_images[doc_id]

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