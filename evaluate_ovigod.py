#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate fine-tuned Rex-Omni model on OV-IGOD test set

Usage:
    python evaluate_ovigod.py --checkpoint finetuning/work_dirs/ovigod_sft-1000/checkpoint-93 --max_samples 100
"""

import argparse
import json
import os
import re
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import numpy as np

from rex_omni import RexOmniWrapper


def parse_target_bboxes(target_string):
    """Extract all bbox coordinates from target string (normalized to [0,999])"""
    coords = re.findall(r'<loc_(\d+)>', target_string)
    coords = [int(c) for c in coords]
    
    # Every 4 coordinates form one bbox
    bboxes = []
    for i in range(0, len(coords), 4):
        if i + 3 < len(coords):
            bboxes.append([coords[i], coords[i+1], coords[i+2], coords[i+3]])
    
    return bboxes


def compute_iou(box1, box2):
    """Compute IoU of two bboxes"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Compute intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # Compute union
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def denormalize_box(norm_box, img_width, img_height):
    """Convert normalized coordinates [0,999] back to absolute coordinates"""
    x0 = (norm_box[0] / 999.0) * img_width
    y0 = (norm_box[1] / 999.0) * img_height
    x1 = (norm_box[2] / 999.0) * img_width
    y1 = (norm_box[3] / 999.0) * img_height
    return [x0, y0, x1, y1]


def evaluate_sample(gt_boxes, pred_boxes, iou_threshold=0.5):
    """
    Evaluate a single sample
    Returns: (true_positives, false_positives, false_negatives)
    """
    if len(gt_boxes) == 0 and len(pred_boxes) == 0:
        return 0, 0, 0
    
    if len(gt_boxes) == 0:
        return 0, len(pred_boxes), 0
    
    if len(pred_boxes) == 0:
        return 0, 0, len(gt_boxes)
    
    # Match predicted boxes with ground truth boxes
    matched_gt = set()
    tp = 0
    
    for pred_box in pred_boxes:
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue
            
            iou = compute_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_iou >= iou_threshold:
            tp += 1
            matched_gt.add(best_gt_idx)
    
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - len(matched_gt)
    
    return tp, fp, fn


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on OV-IGOD test set")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to fine-tuned model checkpoint"
    )
    parser.add_argument(
        "--test_json",
        type=str,
        default="/workspace/hairong/data/ov-igod-dataset/test.json",
        help="Path to test set JSON file"
    )
    parser.add_argument(
        "--image_root",
        type=str,
        default="/workspace/hairong/data/ov-igod-dataset/sunrgbd_jpgs",
        help="Root directory of images"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to test (None=all)"
    )
    parser.add_argument(
        "--iou_threshold",
        type=float,
        default=0.5,
        help="IoU threshold (default: 0.5)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="evaluation_results.json",
        help="Path to save evaluation results"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="transformers",
        choices=["transformers", "vllm"],
        help="Inference backend"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("OV-IGOD Test Set Evaluation")
    print("="*80)
    
    # Load test set
    print(f"\nLoading test set: {args.test_json}")
    with open(args.test_json, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    # Sample test_ids uniformly
    all_test_ids = list(test_data.keys())
    if args.max_samples and args.max_samples < len(all_test_ids):
        # ⭐ Uniform sampling: take samples evenly spaced across the dataset
        step = len(all_test_ids) / args.max_samples
        test_ids = [all_test_ids[int(i * step)] for i in range(args.max_samples)]
        print(f"Uniformly sampled {len(test_ids)} from {len(all_test_ids)} test samples")
    else:
        test_ids = all_test_ids
        print(f"Loaded all {len(test_ids)} test samples")
    
    # Load model
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
    
    # Evaluation metrics
    total_tp = 0
    total_fp = 0
    total_fn = 0
    sample_results = []
    
    print(f"\nStarting evaluation...")
    print(f"IoU threshold: {args.iou_threshold}")
    
    # Evaluate each sample
    for img_id in tqdm(test_ids, desc="Evaluation progress"):
        sample = test_data[img_id]
        image_path = os.path.join(args.image_root, f"{img_id}.jpg")
        
        # Check if image exists
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue
        
        try:
            # Load image
            image = Image.open(image_path).convert("RGB")
            img_width, img_height = image.size
            
            # Prepare ground truth
            gt_affordances = {}
            for bbox_item in sample['bboxes']:
                affordance = bbox_item['affordance']
                target_bboxes = parse_target_bboxes(bbox_item['target'])
                
                # Convert to absolute coordinates
                abs_boxes = [denormalize_box(box, img_width, img_height) for box in target_bboxes]
                
                if affordance not in gt_affordances:
                    gt_affordances[affordance] = []
                gt_affordances[affordance].extend(abs_boxes)
            
            # Prepare detection categories (all affordances present)
            categories = list(gt_affordances.keys())
            
            # Inference
            results = rex.inference(images=image, task="detection", categories=categories)
            result = results[0]
            predictions = result["extracted_predictions"]
            
            # Evaluate each affordance
            sample_tp = 0
            sample_fp = 0
            sample_fn = 0
            
            for affordance in categories:
                gt_boxes = gt_affordances.get(affordance, [])
                pred_boxes = [pred['coords'] for pred in predictions.get(affordance, [])]
                
                tp, fp, fn = evaluate_sample(gt_boxes, pred_boxes, args.iou_threshold)
                sample_tp += tp
                sample_fp += fp
                sample_fn += fn
            
            total_tp += sample_tp
            total_fp += sample_fp
            total_fn += sample_fn
            
            # Record sample results
            sample_results.append({
                "image_id": img_id,
                "tp": sample_tp,
                "fp": sample_fp,
                "fn": sample_fn,
                "gt_count": sum(len(boxes) for boxes in gt_affordances.values()),
                "pred_count": sum(len(preds) for preds in predictions.values())
            })
            
        except Exception as e:
            print(f"\nError processing sample {img_id}: {e}")
            continue
    
    # Compute final metrics
    print("\n" + "="*80)
    print("Evaluation Results")
    print("="*80)
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nNumber of samples: {len(sample_results)}")
    print(f"True Positives (TP): {total_tp}")
    print(f"False Positives (FP): {total_fp}")
    print(f"False Negatives (FN): {total_fn}")
    print(f"\nPrecision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall: {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1 Score: {f1:.4f} ({f1*100:.2f}%)")
    
    # Save results
    results_dict = {
        "checkpoint": args.checkpoint,
        "test_samples": len(sample_results),
        "iou_threshold": args.iou_threshold,
        "metrics": {
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "tp": int(total_tp),
            "fp": int(total_fp),
            "fn": int(total_fn)
        },
        "sample_results": sample_results
    }
    
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)
    
    print(f"\nDetailed results saved to: {args.output_file}")
    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()

