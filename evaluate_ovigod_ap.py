#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate fine-tuned Rex-Omni model on OV-IGOD test set - Calculate AP metrics

Usage:
    python evaluate_ovigod_ap.py --checkpoint finetuning/work_dirs/ovigod_sft-1000/checkpoint-93 --max_samples 100 --batch_size 4
"""

import argparse
import json
import os
import re
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp

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


def compute_ap(recalls, precisions):
    """
    Compute Average Precision (AP)
    Using 11-point interpolation (VOC style) or all-point interpolation
    """
    # Add sentinel values
    mrec = np.concatenate(([0.], recalls, [1.]))
    mpre = np.concatenate(([0.], precisions, [0.]))
    
    # Compute precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    
    # Find points where recall changes
    i = np.where(mrec[1:] != mrec[:-1])[0]
    
    # Compute area under curve
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    
    return ap


def evaluate_at_iou_threshold(all_predictions, all_ground_truths, iou_threshold):
    """
    Evaluate all predictions at a specific IoU threshold
    
    Returns:
        ap: Average Precision
    """
    # Collect all predictions and ground truths
    all_pred_boxes = []
    all_gt_boxes = []
    
    for sample_id in all_ground_truths.keys():
        gt_boxes = all_ground_truths[sample_id]
        pred_boxes = all_predictions.get(sample_id, [])
        
        # Add match flag to each GT
        for gt_box in gt_boxes:
            all_gt_boxes.append({
                'sample_id': sample_id,
                'box': gt_box,
                'matched': False
            })
        
        # Add confidence to each prediction (we don't have confidence, use 1.0)
        for pred_box in pred_boxes:
            all_pred_boxes.append({
                'sample_id': sample_id,
                'box': pred_box,
                'confidence': 1.0  # Model doesn't output confidence, assume 1.0
            })
    
    if len(all_gt_boxes) == 0:
        return 0.0
    
    if len(all_pred_boxes) == 0:
        return 0.0
    
    # Sort predictions by confidence (all 1.0 here, order unchanged)
    all_pred_boxes.sort(key=lambda x: x['confidence'], reverse=True)
    
    # Compute TP and FP
    tp = np.zeros(len(all_pred_boxes))
    fp = np.zeros(len(all_pred_boxes))
    
    # Create index for GT of each sample
    gt_by_sample = {}
    for i, gt in enumerate(all_gt_boxes):
        sample_id = gt['sample_id']
        if sample_id not in gt_by_sample:
            gt_by_sample[sample_id] = []
        gt_by_sample[sample_id].append(i)
    
    for pred_idx, pred in enumerate(all_pred_boxes):
        sample_id = pred['sample_id']
        pred_box = pred['box']
        
        # Find all GTs for this sample
        gt_indices = gt_by_sample.get(sample_id, [])
        
        max_iou = 0
        max_gt_idx = -1
        
        for gt_idx in gt_indices:
            gt = all_gt_boxes[gt_idx]
            if gt['matched']:
                continue
            
            iou = compute_iou(pred_box, gt['box'])
            if iou > max_iou:
                max_iou = iou
                max_gt_idx = gt_idx
        
        if max_iou >= iou_threshold and max_gt_idx != -1:
            tp[pred_idx] = 1
            all_gt_boxes[max_gt_idx]['matched'] = True
        else:
            fp[pred_idx] = 1
    
    # Compute cumulative TP and FP
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    # Compute precision and recall
    recalls = tp_cumsum / len(all_gt_boxes)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
    
    # Compute AP
    ap = compute_ap(recalls, precisions)
    
    return ap


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on OV-IGOD test set - AP metrics")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to fine-tuned model checkpoint"
    )
    parser.add_argument(
        "--test_json",
        type=str,
        default="/home/hairong/hairong/code/ov-igod-dataset/test.json",
        help="Path to test set JSON file"
    )
    parser.add_argument(
        "--image_root",
        type=str,
        default="/home/hairong/hairong/code/ov-igod-dataset/sunrgbd_jpgs",
        help="Root directory of images"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to test (None=all)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="evaluation_ap_results.json",
        help="Path to save evaluation results"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="vllm",
        choices=["transformers", "vllm"],
        help="Inference backend"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for inference (use >1 for faster processing)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of worker processes for data loading (use >1 for parallel processing)"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("OV-IGOD Test Set Evaluation - AP Metrics")
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
    
    # Collect all predictions and ground truths
    all_predictions = {}  # sample_id -> [boxes]
    all_ground_truths = {}  # sample_id -> [boxes]
    
    print(f"\nStarting inference...")
    print(f"Batch size: {args.batch_size}, Num workers: {args.num_workers}")
    
    # Helper function to load a single sample
    def load_sample(img_id):
        """Load and prepare a single sample (for parallel loading)"""
        try:
            sample = test_data[img_id]
            image_path = os.path.join(args.image_root, f"{img_id}.jpg")
            
            if not os.path.exists(image_path):
                return None, f"Image not found: {image_path}"
            
            # Load image
            image = Image.open(image_path).convert("RGB")
            img_width, img_height = image.size
            
            # Prepare ground truth
            gt_boxes = []
            gt_affordances = {}
            for bbox_item in sample['bboxes']:
                affordance = bbox_item['affordance']
                target_bboxes = parse_target_bboxes(bbox_item['target'])
                
                # Convert to absolute coordinates
                abs_boxes = [denormalize_box(box, img_width, img_height) for box in target_bboxes]
                gt_boxes.extend(abs_boxes)
                
                if affordance not in gt_affordances:
                    gt_affordances[affordance] = []
                gt_affordances[affordance].extend(abs_boxes)
            
            # Prepare detection categories
            categories = list(gt_affordances.keys())
            
            return {
                'img_id': img_id,
                'image': image,
                'categories': categories,
                'gt_boxes': gt_boxes,
                'img_width': img_width,
                'img_height': img_height
            }, None
            
        except Exception as e:
            return None, f"Error loading {img_id}: {e}"
    
    # Process in batches for efficiency
    batch_size = args.batch_size
    num_batches = (len(test_ids) + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="Inference progress"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(test_ids))
        batch_ids = test_ids[start_idx:end_idx]
        
        # Prepare batch data with parallel loading
        batch_images = []
        batch_categories = []
        batch_metadata = []
        
        # Use multithreading for parallel data loading if num_workers > 1
        # ThreadPoolExecutor is better for I/O-bound tasks like image loading
        if args.num_workers > 1:
            with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
                futures = {executor.submit(load_sample, img_id): img_id for img_id in batch_ids}
                
                for future in as_completed(futures):
                    result, error = future.result()
                    if error:
                        print(f"\nWarning: {error}")
                        continue
                    
                    if result:
                        all_ground_truths[result['img_id']] = result['gt_boxes']
                        batch_images.append(result['image'])
                        batch_categories.append(result['categories'])
                        batch_metadata.append({
                            'img_id': result['img_id'],
                            'img_width': result['img_width'],
                            'img_height': result['img_height']
                        })
        else:
            # Sequential loading (num_workers <= 1)
            for img_id in batch_ids:
                result, error = load_sample(img_id)
                if error:
                    print(f"\nWarning: {error}")
                    continue
                
                if result:
                    all_ground_truths[result['img_id']] = result['gt_boxes']
                    batch_images.append(result['image'])
                    batch_categories.append(result['categories'])
                    batch_metadata.append({
                        'img_id': result['img_id'],
                        'img_width': result['img_width'],
                        'img_height': result['img_height']
                    })
        
        # Batch inference
        if len(batch_images) > 0:
            try:
                results = rex.inference(
                    images=batch_images, 
                    task="detection", 
                    categories=batch_categories
                )
                
                # Process batch results
                for result, metadata in zip(results, batch_metadata):
                    img_id = metadata['img_id']
                    categories = batch_categories[batch_metadata.index(metadata)]
                    predictions = result["extracted_predictions"]
                    
                    # Collect all predicted boxes (filter out incorrectly formatted boxes)
                    pred_boxes = []
                    for affordance in categories:
                        for pred in predictions.get(affordance, []):
                            coords = pred['coords']
                            # Validate bbox format: must have 4 coordinates
                            if isinstance(coords, (list, tuple)) and len(coords) == 4:
                                pred_boxes.append(coords)
                            else:
                                print(f"Warning: Sample {img_id} has malformed prediction box: {coords}")
                    
                    all_predictions[img_id] = pred_boxes
                    
            except Exception as e:
                print(f"\nError processing batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Compute AP metrics
    print("\nComputing AP metrics...")
    
    # AP@50
    ap50 = evaluate_at_iou_threshold(all_predictions, all_ground_truths, 0.5)
    
    # AP@75
    ap75 = evaluate_at_iou_threshold(all_predictions, all_ground_truths, 0.75)
    
    # AP@50:95 (IoU from 0.5 to 0.95, step 0.05)
    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    aps = []
    for iou_thresh in iou_thresholds:
        ap = evaluate_at_iou_threshold(all_predictions, all_ground_truths, iou_thresh)
        aps.append(ap)
    
    ap_50_95 = np.mean(aps)
    
    # Print results
    print("\n" + "="*80)
    print("Evaluation Results")
    print("="*80)
    
    print(f"\nNumber of samples: {len(all_ground_truths)}")
    print(f"Total GT count: {sum(len(boxes) for boxes in all_ground_truths.values())}")
    print(f"Total predictions: {sum(len(boxes) for boxes in all_predictions.values())}")
    
    print(f"\n{'Metric':<15} {'Value':<10}")
    print("-" * 25)
    print(f"{'AP@50':<15} {ap50:.4f} ({ap50*100:.2f}%)")
    print(f"{'AP@75':<15} {ap75:.4f} ({ap75*100:.2f}%)")
    print(f"{'AP@50:95':<15} {ap_50_95:.4f} ({ap_50_95*100:.2f}%)")
    
    # Save results
    results_dict = {
        "checkpoint": args.checkpoint,
        "test_samples": len(all_ground_truths),
        "total_gt": sum(len(boxes) for boxes in all_ground_truths.values()),
        "total_predictions": sum(len(boxes) for boxes in all_predictions.values()),
        "metrics": {
            "AP@50": float(ap50),
            "AP@75": float(ap75),
            "AP@50:95": float(ap_50_95)
        },
        "detailed_aps": {
            f"AP@{iou:.2f}": float(ap) 
            for iou, ap in zip(iou_thresholds, aps)
        }
    }
    
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)
    
    print(f"\nDetailed results saved to: {args.output_file}")
    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()

