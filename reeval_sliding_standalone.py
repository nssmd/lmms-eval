#!/usr/bin/env python3
"""
Re-evaluate sliding puzzle results with updated evaluation logic.
Standalone version without complex imports.
"""

import json
import re
import sys


def sliding_process_results_updated(doc, results):
    """Updated sliding process results with direction swap."""
    result_raw = results[0] if results else ""

    # Handle case where result is a JSON string
    result_text = ""
    if isinstance(result_raw, str):
        try:
            parsed_result = json.loads(result_raw)
            if isinstance(parsed_result, dict) and "text" in parsed_result:
                result_text = parsed_result["text"]
            else:
                result_text = result_raw
        except (json.JSONDecodeError, TypeError):
            result_text = result_raw
    else:
        result_text = str(result_raw)

    # Parse predicted moves from <ANSWER_JSON>
    pred_moves = []
    matches = list(re.finditer(
        r"<ANSWER_JSON>\s*(\[.*?\])\s*</ANSWER_JSON>",
        result_text,
        re.DOTALL | re.IGNORECASE
    ))
    if matches:
        try:
            moves_data = json.loads(matches[-1].group(1))
            pred_moves = [str(m).strip().lower() for m in moves_data]
        except:
            pass

    # Ground truth moves
    gt_moves_str = doc.get("steps_words", "[]")
    gt_moves = json.loads(gt_moves_str) if isinstance(gt_moves_str, str) else gt_moves_str
    gt_moves = [str(m).lower() for m in gt_moves]

    # Convert ground truth moves: swap up<->down, left<->right
    direction_map = {
        "up": "down",
        "down": "up",
        "left": "right",
        "right": "left"
    }
    gt_moves = [direction_map.get(m, m) for m in gt_moves]

    # Text evaluation
    text_exact = 1 if pred_moves == gt_moves else 0
    text_frame_acc = (
        sum(1 for p, g in zip(pred_moves, gt_moves) if p == g) / len(gt_moves)
        if gt_moves else 0.0
    )

    return {
        "sliding_text_exact": text_exact,
        "sliding_text_frame_acc": text_frame_acc,
    }


def reeval_sliding_results(jsonl_path):
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

            # Reconstruct doc and results
            doc = {
                "steps_words": data["target"]
            }
            results_list = data["filtered_resps"]

            # Re-evaluate with updated logic
            metrics = sliding_process_results_updated(doc, results_list)

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
        print("Usage: python reeval_sliding_standalone.py <path_to_jsonl>")
        sys.exit(1)

    jsonl_path = sys.argv[1]
    reeval_sliding_results(jsonl_path)
