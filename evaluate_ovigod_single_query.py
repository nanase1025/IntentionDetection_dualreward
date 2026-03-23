#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate fine-tuned Rex-Omni model on OV-IGOD test set.

Follows the same evaluation logic as evaluate_intention_three_datasets.py:
  - Single affordance query per sample (matches training prompt format)
  - Accuracy-based AP (not PR-curve AP)
  - Mean IoU metric

Metrics reported: IoU (mean), AP@50, AP@75, AP@50:95

Usage:
    python evaluate_ovigod_single_query.py \
        --checkpoint finetuning/work_dirs/ovigod_sft/checkpoint-93 \
        --max_samples 100

    python evaluate_ovigod_single_query.py \
        --checkpoint IDEA-Research/Rex-Omni \
        --backend vllm --max_samples 200
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from rex_omni import RexOmniWrapper


def compute_iou(box1, box2):
    """Compute IoU of two bboxes (absolute coordinates [x0, y0, x1, y1])"""
    try:
        if isinstance(box1, (list, tuple)) and len(box1) == 4:
            x1_min, y1_min, x1_max, y1_max = (
                float(box1[0]), float(box1[1]), float(box1[2]), float(box1[3])
            )
        else:
            return 0.0

        if isinstance(box2, (list, tuple)) and len(box2) == 4:
            x2_min, y2_min, x2_max, y2_max = (
                float(box2[0]), float(box2[1]), float(box2[2]), float(box2[3])
            )
        else:
            return 0.0
    except (TypeError, ValueError):
        return 0.0

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area


def evaluate_at_iou_threshold(all_predictions, all_ground_truths, iou_threshold):
    """
    Per-sample accuracy at a given IoU threshold.
    A sample counts as TP if the best IoU between any pred-GT pair >= threshold.
    AP = TP_count / total_samples  (same metric as evaluate_intention_three_datasets.py)
    """
    if len(all_ground_truths) == 0:
        return 0.0

    tp_count = 0
    total_samples = len(all_ground_truths)

    for sample_id, gt_boxes in all_ground_truths.items():
        pred_boxes = all_predictions.get(sample_id, [])
        if len(gt_boxes) == 0:
            continue

        best_iou = 0.0
        for pred_box in pred_boxes:
            for gt_box in gt_boxes:
                iou = compute_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou

        if best_iou >= iou_threshold:
            tp_count += 1

    return tp_count / total_samples if total_samples > 0 else 0.0


def load_ovigod_samples(json_path, image_root):
    """
    Load OV-IGOD dataset and flatten to per-(image, affordance) samples.
    Groups bboxes by affordance so that one query has all GT boxes for that affordance.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    samples = []
    for img_id, entry in raw_data.items():
        image_path = os.path.join(image_root, f"{img_id}.jpg")
        if not os.path.exists(image_path):
            continue

        affordance_groups = defaultdict(list)
        for bbox_item in entry["bboxes"]:
            affordance_groups[bbox_item["affordance"]].append(bbox_item["bbox"])

        for affordance, bboxes in affordance_groups.items():
            samples.append({
                "img_id": img_id,
                "image_path": image_path,
                "affordance": affordance,
                "gt_boxes": bboxes,
            })

    return samples


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate model on OV-IGOD test set (single-query, accuracy-based AP)"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to fine-tuned model checkpoint",
    )
    parser.add_argument(
        "--test_json", type=str,
        default="/home/hairong/hairong/code/ov-igod-dataset/test.json",
        help="Path to OV-IGOD test.json",
    )
    parser.add_argument(
        "--image_root", type=str,
        default="/home/hairong/hairong/code/ov-igod-dataset/sunrgbd_jpgs",
        help="Root directory of images",
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Max number of (image, affordance) samples to evaluate (None=all)",
    )
    parser.add_argument(
        "--output_file", type=str,
        default="ours_sft_evaluation_ovigod_single_query_results.json",
        help="Path to save evaluation results",
    )
    parser.add_argument(
        "--backend", type=str, default="vllm",
        choices=["transformers", "vllm"],
        help="Inference backend",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("OV-IGOD Evaluation (Single-Query, Accuracy-based AP)")
    print("=" * 80)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print(f"\nLoading test set: {args.test_json}")
    all_samples = load_ovigod_samples(args.test_json, args.image_root)
    print(f"Total (image, affordance) samples: {len(all_samples)}")

    if args.max_samples and args.max_samples < len(all_samples):
        step = len(all_samples) / args.max_samples
        all_samples = [all_samples[int(i * step)] for i in range(args.max_samples)]
        print(f"Uniformly sampled {len(all_samples)} samples")

    unique_images = len(set(s["img_id"] for s in all_samples))
    print(f"Covering {unique_images} unique images")

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    print(f"\nLoading model: {args.checkpoint}")
    rex = RexOmniWrapper(
        model_path=args.checkpoint,
        backend=args.backend,
        max_tokens=2048,
        temperature=0.0,
        top_p=0.05,
        top_k=1,
        repetition_penalty=1.05,
    )
    print("Model loaded successfully")

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    predictions = {}
    ground_truths = {}
    sample_ious = {}
    iou_list = []

    print(f"\nStarting evaluation...")

    for idx, sample in enumerate(tqdm(all_samples, desc="OV-IGOD")):
        sample_key = f"{sample['img_id']}_{idx}"
        gt_boxes = sample["gt_boxes"]
        affordance = sample["affordance"]

        try:
            image = Image.open(sample["image_path"]).convert("RGB")

            results = rex.inference(
                images=image,
                task="detection",
                categories=[affordance],
            )

            pred_boxes = []
            extracted = results[0]["extracted_predictions"]
            matched_preds = extracted.get(affordance, [])
            if not matched_preds:
                norm_aff = affordance.strip().rstrip(".")
                for key, preds in extracted.items():
                    if key.strip().rstrip(".") == norm_aff:
                        matched_preds = preds
                        break
            for pred in matched_preds:
                coords = pred["coords"]
                if isinstance(coords, (list, tuple)) and len(coords) == 4:
                    pred_boxes.append([float(c) for c in coords])

            predictions[sample_key] = pred_boxes
            ground_truths[sample_key] = gt_boxes

            best_iou = 0.0
            if pred_boxes and gt_boxes:
                for pb in pred_boxes:
                    for gb in gt_boxes:
                        iou = compute_iou(pb, gb)
                        if iou > best_iou:
                            best_iou = iou

            iou_list.append(best_iou)
            sample_ious[sample_key] = {
                "img_id": sample["img_id"],
                "affordance": affordance,
                "iou": float(best_iou),
                "gt_boxes": gt_boxes,
                "pred_boxes": pred_boxes,
            }

        except Exception as e:
            print(f"\nError processing {sample['img_id']} / {affordance[:40]}: {e}")
            predictions[sample_key] = []
            ground_truths[sample_key] = gt_boxes
            iou_list.append(0.0)
            sample_ious[sample_key] = {
                "img_id": sample["img_id"],
                "affordance": affordance,
                "iou": 0.0,
                "gt_boxes": gt_boxes,
                "pred_boxes": [],
            }

    # ------------------------------------------------------------------
    # Compute metrics
    # ------------------------------------------------------------------
    mean_iou = float(np.mean(iou_list)) if iou_list else 0.0
    ap50 = evaluate_at_iou_threshold(predictions, ground_truths, 0.5)
    ap75 = evaluate_at_iou_threshold(predictions, ground_truths, 0.75)

    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    aps = [evaluate_at_iou_threshold(predictions, ground_truths, t) for t in iou_thresholds]
    ap_50_95 = float(np.mean(aps))

    # ------------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("OV-IGOD Evaluation Results")
    print("=" * 80)
    print(f"\n  Samples:   {len(iou_list)}")
    print(f"  Mean IoU:  {mean_iou:.4f} ({mean_iou * 100:.2f}%)")
    print(f"  AP@50:     {ap50:.4f} ({ap50 * 100:.2f}%)")
    print(f"  AP@75:     {ap75:.4f} ({ap75 * 100:.2f}%)")
    print(f"  AP@50:95:  {ap_50_95:.4f} ({ap_50_95 * 100:.2f}%)")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    results_dict = {
        "checkpoint": args.checkpoint,
        "dataset": "ov-igod",
        "num_samples": len(iou_list),
        "num_unique_images": unique_images,
        "metrics": {
            "mean_iou": mean_iou,
            "AP@50": float(ap50),
            "AP@75": float(ap75),
            "AP@50:95": ap_50_95,
        },
        "detailed_aps": {
            f"AP@{t:.2f}": float(a) for t, a in zip(iou_thresholds, aps)
        },
        "sample_results": sample_ious,
    }

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)

    print(f"\nDetailed results saved to: {args.output_file}")
    print("Evaluation completed!")


if __name__ == "__main__":
    main()
