#!/usr/bin/env python3
"""
Re-evaluate sliding puzzle results with updated evaluation logic.
"""

import json
import sys
from pathlib import Path

# Import the updated sliding_process_results function
sys.path.insert(0, str(Path(__file__).parent))
from lmms_eval.tasks.uni_mmmu.utils import sliding_process_results


def reeval_sliding_results(jsonl_path: str):
    """Re-evaluate sliding puzzle results from saved jsonl file."""

    results = []
    total_exact = 0
    total_frame_acc = 0
    count = 0

    print(f"Reading results from: {jsonl_path}")
    print("=" * 80)

    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line)

            # Reconstruct doc and results for sliding_process_results
            doc = {
                "steps_words": data["target"]
            }
            results_list = data["filtered_resps"]

            # Re-evaluate with updated logic
            metrics = sliding_process_results(doc, results_list)

            # Store results
            results.append({
                "doc_id": data["doc_id"],
                "old_exact": data["sliding_text_exact"],
                "new_exact": metrics["sliding_text_exact"],
                "old_frame_acc": data["sliding_text_frame_acc"],
                "new_frame_acc": metrics["sliding_text_frame_acc"],
            })

            total_exact += metrics["sliding_text_exact"]
            total_frame_acc += metrics["sliding_text_frame_acc"]
            count += 1

    # Print summary
    print(f"\nProcessed {count} samples")
    print("=" * 80)
    print("\nOLD METRICS (before direction swap):")
    old_exact_avg = sum(r["old_exact"] for r in results) / count
    old_frame_avg = sum(r["old_frame_acc"] for r in results) / count
    print(f"  Exact Match:     {old_exact_avg:.4f} ({sum(r['old_exact'] for r in results)}/{count})")
    print(f"  Frame Accuracy:  {old_frame_avg:.4f}")

    print("\nNEW METRICS (after direction swap):")
    new_exact_avg = total_exact / count
    new_frame_avg = total_frame_acc / count
    print(f"  Exact Match:     {new_exact_avg:.4f} ({int(total_exact)}/{count})")
    print(f"  Frame Accuracy:  {new_frame_avg:.4f}")

    print("\nIMPROVEMENT:")
    print(f"  Exact Match:     {(new_exact_avg - old_exact_avg):.4f} ({(new_exact_avg - old_exact_avg) * 100:.2f}%)")
    print(f"  Frame Accuracy:  {(new_frame_avg - old_frame_avg):.4f} ({(new_frame_avg - old_frame_avg) * 100:.2f}%)")
    print("=" * 80)

    # Show samples with changed results
    changed = [r for r in results if r["old_exact"] != r["new_exact"]]
    if changed:
        print(f"\n{len(changed)} samples changed from incorrect to correct:")
        for r in changed[:10]:  # Show first 10
            print(f"  doc_id {r['doc_id']}: exact {r['old_exact']} -> {r['new_exact']}, "
                  f"frame_acc {r['old_frame_acc']:.3f} -> {r['new_frame_acc']:.3f}")
        if len(changed) > 10:
            print(f"  ... and {len(changed) - 10} more")

    return results


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python reeval_sliding.py <path_to_jsonl>")
        sys.exit(1)

    jsonl_path = sys.argv[1]
    reeval_sliding_results(jsonl_path)
