#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate fine-tuned Rex-Omni model on three intention detection datasets

This script evaluates the model on:
- COCO Outdoor test set
- ScanNet test set
- EgoObject test set (from egoobject-intention-eval)

Metrics reported: IoU (mean), AP@50, AP@75, AP@50:95

Usage:
    python evaluate_intention_three_datasets.py \
        --checkpoint finetuning/work_dirs/intention_grpo_filtered_epoch1/global_step_556/actor/huggingface \
        --max_samples 100
"""

import argparse
import json
import base64
import io
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import numpy as np
from collections import defaultdict

from rex_omni import RexOmniWrapper


def compute_iou(box1, box2):
    """Compute IoU of two bboxes (absolute coordinates [x0, y0, x1, y1])"""
    try:
        if isinstance(box1, (list, tuple)) and len(box1) == 4:
            x1_min, y1_min, x1_max, y1_max = float(box1[0]), float(box1[1]), float(box1[2]), float(box1[3])
        else:
            print(f"Warning: Invalid box1 format: {box1}")
            return 0.0
            
        if isinstance(box2, (list, tuple)) and len(box2) == 4:
            x2_min, y2_min, x2_max, y2_max = float(box2[0]), float(box2[1]), float(box2[2]), float(box2[3])
        else:
            print(f"Warning: Invalid box2 format: {box2}")
            return 0.0
    except (TypeError, ValueError) as e:
        print(f"Error converting box coordinates: {e}")
        return 0.0
    
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


def compute_ap(recalls, precisions):
    """Compute Average Precision (AP) using all-point interpolation"""
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
a

def evaluate_at_iou_threshold(all_predictions, all_ground_truths, iou_threshold):
    """
    Evaluate predictions at a specific IoU threshold using one-to-one matching.
    
    For intention detection: each sample may have one or more GT boxes.
    We select the prediction with highest IoU against ANY GT box for each sample.
    AP = Accuracy = (number of samples with max IoU >= threshold) / total_samples
    
    Args:
        all_predictions: Dict[sample_id -> List[boxes]]
        all_ground_truths: Dict[sample_id -> List[boxes]]
        iou_threshold: IoU threshold
    
    Returns:
        ap: Average Precision (= Accuracy in one-to-one case)
    """
    if len(all_ground_truths) == 0:
        return 0.0
    
    tp_count = 0
    total_samples = len(all_ground_truths)
    
    for sample_id, gt_boxes in all_ground_truths.items():
        pred_boxes = all_predictions.get(sample_id, [])
        
        if len(gt_boxes) == 0:
            continue
        
        # Find the best IoU across all GT boxes and all predictions
        best_iou = 0.0
        if len(pred_boxes) > 0:
            for pred_box in pred_boxes:
                for gt_box in gt_boxes:
                    iou = compute_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
        
        # Count as TP if best IoU >= threshold
        if best_iou >= iou_threshold:
            tp_count += 1
    
    # AP = Accuracy for one-to-one matching
    ap = tp_count / total_samples if total_samples > 0 else 0.0
    
    return ap


def load_tsv_samples(img_tsv_file, ann_tsv_file, ann_lineidx_file):
    """Load samples from TSV files (for COCO, ScanNet, and EgoObject)"""
    samples = []
    
    with open(ann_lineidx_file, 'r') as f:
        anno_line_offsets = [int(line.strip()) for line in f]
    
    for idx in range(len(anno_line_offsets)):
        # Read annotation
        with open(ann_tsv_file, 'rb') as f_ann:
            f_ann.seek(anno_line_offsets[idx])
            line = f_ann.readline().decode('utf-8').strip()
            img_byte_offset_str, ann_json = line.split('\t')
            img_byte_offset = int(img_byte_offset_str)
            annotation = json.loads(ann_json)
        
        # Read image
        with open(img_tsv_file, 'rb') as f_img:
            f_img.seek(img_byte_offset)
            img_line = f_img.readline().decode('utf-8').strip()
            sample_id, img_base64 = img_line.split('\t')
            img_bytes = base64.b64decode(img_base64)
            image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        
        samples.append({
            'sample_id': sample_id,
            'image': image,
            'annotation': annotation
        })
    
    return samples


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on three intention detection datasets")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to fine-tuned model checkpoint"
    )
    parser.add_argument(
        "--coco_test_tsv",
        type=str,
        default="/home/hairong/hairong/data/intention_datasets_tsv_fixed",
        help="Directory containing COCO test TSV files"
    )
    parser.add_argument(
        "--scannet_test_tsv",
        type=str,
        default="/home/hairong/hairong/data/intention_datasets_tsv_fixed",
        help="Directory containing ScanNet test TSV files"
    )
    parser.add_argument(
        "--egoobject_test_tsv",
        type=str,
        default="/home/hairong/hairong/data/intention_datasets_tsv_fixed",
        help="Directory containing EgoObject test TSV files"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to test per dataset (None=all)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="evaluation_three_datasets_results.json",
        help="Path to save evaluation results"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="vllm",
        choices=["transformers", "vllm"],
        help="Inference backend"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("Three Datasets Evaluation - IoU, AP@50, AP@75, AP@50:95")
    print("="*80)
    
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
    
    # Store results for each dataset
    dataset_results = {}
    
    # ========================================
    # 1. Evaluate COCO Outdoor
    # ========================================
    print("\n" + "="*80)
    print("Evaluating COCO Outdoor Test Set")
    print("="*80)
    
    coco_dir = Path(args.coco_test_tsv)
    coco_samples = load_tsv_samples(
        coco_dir / "coco_outdoor_test.images.tsv",
        coco_dir / "coco_outdoor_test.annotations.tsv",
        coco_dir / "coco_outdoor_test.annotations.tsv.lineidx"
    )
    
    if args.max_samples and args.max_samples < len(coco_samples):
        step = len(coco_samples) / args.max_samples
        coco_samples = [coco_samples[int(i * step)] for i in range(args.max_samples)]
    
    print(f"Total samples: {len(coco_samples)}")
    
    coco_predictions = {}
    coco_ground_truths = {}
    coco_ious = []
    coco_sample_ious = {}  # Store IoU for each sample
    
    for sample in tqdm(coco_samples, desc="COCO Outdoor"):
        sample_id = sample['sample_id']
        image = sample['image']
        annotation = sample['annotation']
        
        # Get ground truth
        gt_boxes_data = annotation['boxes']
        gt_boxes = [box['bbox'] for box in gt_boxes_data]
        phrase = gt_boxes_data[0].get('phrase', 'object')
        
        # Run inference
        try:
            results = rex.inference(
                images=image,
                task="detection",
                categories=[phrase]
            )
            
            # Extract predictions
            pred_boxes = []
            for pred in results[0]["extracted_predictions"].get(phrase, []):
                coords = pred['coords']
                if isinstance(coords, (list, tuple)) and len(coords) == 4:
                    pred_boxes.append([float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3])])
            
            coco_predictions[sample_id] = pred_boxes
            coco_ground_truths[sample_id] = gt_boxes
            
            # Compute IoU for this sample
            if pred_boxes and gt_boxes:
                max_iou = 0
                for pred_box in pred_boxes:
                    for gt_box in gt_boxes:
                        iou = compute_iou(pred_box, gt_box)
                        if iou > max_iou:
                            max_iou = iou
                coco_ious.append(max_iou)
                coco_sample_ious[sample_id] = float(max_iou)
            else:
                coco_ious.append(0.0)
                coco_sample_ious[sample_id] = 0.0
        except Exception as e:
            print(f"Error processing {sample_id}: {e}")
            coco_predictions[sample_id] = []
            coco_ground_truths[sample_id] = gt_boxes
            coco_ious.append(0.0)
            coco_sample_ious[sample_id] = 0.0
    
    # Compute metrics for COCO
    coco_mean_iou = np.mean(coco_ious) if coco_ious else 0.0
    coco_ap50 = evaluate_at_iou_threshold(coco_predictions, coco_ground_truths, 0.5)
    coco_ap75 = evaluate_at_iou_threshold(coco_predictions, coco_ground_truths, 0.75)
    
    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    coco_aps = [evaluate_at_iou_threshold(coco_predictions, coco_ground_truths, t) for t in iou_thresholds]
    coco_ap_50_95 = np.mean(coco_aps)
    
    dataset_results['coco_outdoor'] = {
        'num_samples': len(coco_samples),
        'mean_iou': float(coco_mean_iou),
        'AP@50': float(coco_ap50),
        'AP@75': float(coco_ap75),
        'AP@50:95': float(coco_ap_50_95)
    }
    
    print(f"\nCOCO Outdoor Results:")
    print(f"  Mean IoU:  {coco_mean_iou:.4f} ({coco_mean_iou*100:.2f}%)")
    print(f"  AP@50:     {coco_ap50:.4f} ({coco_ap50*100:.2f}%)")
    print(f"  AP@75:     {coco_ap75:.4f} ({coco_ap75*100:.2f}%)")
    print(f"  AP@50:95:  {coco_ap_50_95:.4f} ({coco_ap_50_95*100:.2f}%)")
    
    # ========================================
    # 2. Evaluate ScanNet
    # ========================================
    print("\n" + "="*80)
    print("Evaluating ScanNet Test Set")
    print("="*80)
    
    scannet_dir = Path(args.scannet_test_tsv)
    scannet_samples = load_tsv_samples(
        scannet_dir / "scannet_test.images.tsv",
        scannet_dir / "scannet_test.annotations.tsv",
        scannet_dir / "scannet_test.annotations.tsv.lineidx"
    )
    
    if args.max_samples and args.max_samples < len(scannet_samples):
        step = len(scannet_samples) / args.max_samples
        scannet_samples = [scannet_samples[int(i * step)] for i in range(args.max_samples)]
    
    print(f"Total samples: {len(scannet_samples)}")
    
    scannet_predictions = {}
    scannet_ground_truths = {}
    scannet_ious = []
    scannet_sample_ious = {}  # Store IoU for each sample
    
    for sample in tqdm(scannet_samples, desc="ScanNet"):
        sample_id = sample['sample_id']
        image = sample['image']
        annotation = sample['annotation']
        
        # Get ground truth
        gt_boxes_data = annotation['boxes']
        gt_boxes = [box['bbox'] for box in gt_boxes_data]
        phrase = gt_boxes_data[0].get('phrase', 'object')
        
        # Run inference
        try:
            results = rex.inference(
                images=image,
                task="detection",
                categories=[phrase]
            )
            
            # Extract predictions
            pred_boxes = []
            for pred in results[0]["extracted_predictions"].get(phrase, []):
                coords = pred['coords']
                if isinstance(coords, (list, tuple)) and len(coords) == 4:
                    pred_boxes.append([float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3])])
            
            scannet_predictions[sample_id] = pred_boxes
            scannet_ground_truths[sample_id] = gt_boxes
            
            # Compute IoU for this sample
            if pred_boxes and gt_boxes:
                max_iou = 0
                for pred_box in pred_boxes:
                    for gt_box in gt_boxes:
                        iou = compute_iou(pred_box, gt_box)
                        if iou > max_iou:
                            max_iou = iou
                scannet_ious.append(max_iou)
                scannet_sample_ious[sample_id] = float(max_iou)
            else:
                scannet_ious.append(0.0)
                scannet_sample_ious[sample_id] = 0.0
        except Exception as e:
            print(f"Error processing {sample_id}: {e}")
            scannet_predictions[sample_id] = []
            scannet_ground_truths[sample_id] = gt_boxes
            scannet_ious.append(0.0)
            scannet_sample_ious[sample_id] = 0.0
    
    # Compute metrics for ScanNet
    scannet_mean_iou = np.mean(scannet_ious) if scannet_ious else 0.0
    scannet_ap50 = evaluate_at_iou_threshold(scannet_predictions, scannet_ground_truths, 0.5)
    scannet_ap75 = evaluate_at_iou_threshold(scannet_predictions, scannet_ground_truths, 0.75)
    
    scannet_aps = [evaluate_at_iou_threshold(scannet_predictions, scannet_ground_truths, t) for t in iou_thresholds]
    scannet_ap_50_95 = np.mean(scannet_aps)
    
    dataset_results['scannet'] = {
        'num_samples': len(scannet_samples),
        'mean_iou': float(scannet_mean_iou),
        'AP@50': float(scannet_ap50),
        'AP@75': float(scannet_ap75),
        'AP@50:95': float(scannet_ap_50_95)
    }
    
    print(f"\nScanNet Results:")
    print(f"  Mean IoU:  {scannet_mean_iou:.4f} ({scannet_mean_iou*100:.2f}%)")
    print(f"  AP@50:     {scannet_ap50:.4f} ({scannet_ap50*100:.2f}%)")
    print(f"  AP@75:     {scannet_ap75:.4f} ({scannet_ap75*100:.2f}%)")
    print(f"  AP@50:95:  {scannet_ap_50_95:.4f} ({scannet_ap_50_95*100:.2f}%)")
    
    # ========================================
    # 3. Evaluate EgoObject
    # ========================================
    print("\n" + "="*80)
    print("Evaluating EgoObject Test Set")
    print("="*80)
    
    egoobject_dir = Path(args.egoobject_test_tsv)
    egoobject_samples = load_tsv_samples(
        egoobject_dir / "egoobject_test.images.tsv",
        egoobject_dir / "egoobject_test.annotations.tsv",
        egoobject_dir / "egoobject_test.annotations.tsv.lineidx"
    )
    
    if args.max_samples and args.max_samples < len(egoobject_samples):
        step = len(egoobject_samples) / args.max_samples
        egoobject_samples = [egoobject_samples[int(i * step)] for i in range(args.max_samples)]
    
    print(f"Total samples: {len(egoobject_samples)}")
    
    egoobject_predictions = {}
    egoobject_ground_truths = {}
    egoobject_ious = []
    egoobject_sample_ious = {}  # Store IoU for each sample
    
    for sample in tqdm(egoobject_samples, desc="EgoObject"):
        sample_id = sample['sample_id']
        image = sample['image']
        annotation = sample['annotation']
        
        # Get ground truth
        gt_boxes_data = annotation['boxes']
        gt_boxes = [box['bbox'] for box in gt_boxes_data]
        phrase = gt_boxes_data[0].get('phrase', 'object')
        
        # Run inference
        try:
            results = rex.inference(
                images=image,
                task="detection",
                categories=[phrase]
            )
            
            # Extract predictions
            pred_boxes = []
            for pred in results[0]["extracted_predictions"].get(phrase, []):
                coords = pred['coords']
                if isinstance(coords, (list, tuple)) and len(coords) == 4:
                    pred_boxes.append([float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3])])
            
            egoobject_predictions[sample_id] = pred_boxes
            egoobject_ground_truths[sample_id] = gt_boxes
            
            # Compute IoU for this sample
            if pred_boxes and gt_boxes:
                max_iou = 0
                for pred_box in pred_boxes:
                    for gt_box in gt_boxes:
                        iou = compute_iou(pred_box, gt_box)
                        if iou > max_iou:
                            max_iou = iou
                egoobject_ious.append(max_iou)
                egoobject_sample_ious[sample_id] = float(max_iou)
            else:
                egoobject_ious.append(0.0)
                egoobject_sample_ious[sample_id] = 0.0
        except Exception as e:
            print(f"Error processing {sample_id}: {e}")
            egoobject_predictions[sample_id] = []
            egoobject_ground_truths[sample_id] = gt_boxes
            egoobject_ious.append(0.0)
            egoobject_sample_ious[sample_id] = 0.0
    
    # Compute metrics for EgoObject
    egoobject_mean_iou = np.mean(egoobject_ious) if egoobject_ious else 0.0
    egoobject_ap50 = evaluate_at_iou_threshold(egoobject_predictions, egoobject_ground_truths, 0.5)
    egoobject_ap75 = evaluate_at_iou_threshold(egoobject_predictions, egoobject_ground_truths, 0.75)
    
    egoobject_aps = [evaluate_at_iou_threshold(egoobject_predictions, egoobject_ground_truths, t) for t in iou_thresholds]
    egoobject_ap_50_95 = np.mean(egoobject_aps)
    
    dataset_results['egoobject'] = {
        'num_samples': len(egoobject_samples),
        'mean_iou': float(egoobject_mean_iou),
        'AP@50': float(egoobject_ap50),
        'AP@75': float(egoobject_ap75),
        'AP@50:95': float(egoobject_ap_50_95)
    }
    
    print(f"\nEgoObject Results:")
    print(f"  Mean IoU:  {egoobject_mean_iou:.4f} ({egoobject_mean_iou*100:.2f}%)")
    print(f"  AP@50:     {egoobject_ap50:.4f} ({egoobject_ap50*100:.2f}%)")
    print(f"  AP@75:     {egoobject_ap75:.4f} ({egoobject_ap75*100:.2f}%)")
    print(f"  AP@50:95:  {egoobject_ap_50_95:.4f} ({egoobject_ap_50_95*100:.2f}%)")
    
    # ========================================
    # Summary
    # ========================================
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\n{'Dataset':<20} {'Mean IoU':>12} {'AP@50':>12} {'AP@75':>12} {'AP@50:95':>12}")
    print("-" * 68)
    for dataset_name, metrics in dataset_results.items():
        print(f"{dataset_name:<20} {metrics['mean_iou']:>12.4f} {metrics['AP@50']:>12.4f} {metrics['AP@75']:>12.4f} {metrics['AP@50:95']:>12.4f}")
    
    # Save results
    results_dict = {
        "checkpoint": args.checkpoint,
        "dataset_results": dataset_results,
        "detailed_predictions": {
            "coco_outdoor": {
                sample_id: {
                    "ground_truth": coco_ground_truths[sample_id],
                    "predictions": coco_predictions[sample_id],
                    "iou": coco_sample_ious[sample_id]
                }
                for sample_id in coco_predictions.keys()
            },
            "scannet": {
                sample_id: {
                    "ground_truth": scannet_ground_truths[sample_id],
                    "predictions": scannet_predictions[sample_id],
                    "iou": scannet_sample_ious[sample_id]
                }
                for sample_id in scannet_predictions.keys()
            },
            "egoobject": {
                sample_id: {
                    "ground_truth": egoobject_ground_truths[sample_id],
                    "predictions": egoobject_predictions[sample_id],
                    "iou": egoobject_sample_ious[sample_id]
                }
                for sample_id in egoobject_predictions.keys()
            }
        }
    }
    
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)
    
    print(f"\nDetailed results saved to: {args.output_file}")
    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()

