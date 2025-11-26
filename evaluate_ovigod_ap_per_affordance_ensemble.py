#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate fine-tuned Rex-Omni model on OV-IGOD test set - Per-Affordance AP metrics with Ensemble

This script combines:
1. Per-affordance evaluation (like evaluate_ovigod_ap_per_affordance.py)
2. Temperature ensemble + voting (like evaluate_ovigod_ap_ensemble.py)

Usage:
    python evaluate_ovigod_ap_per_affordance_ensemble.py \
        --checkpoint finetuning/work_dirs/ovigod_sft_5ep \
        --n_samples 5 \
        --vote_threshold 0.4 \
        --temperature 1.0 \
        --max_samples 100
"""

import argparse
import json
import os
import re
import random
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import numpy as np
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    try:
        if isinstance(box1, (list, tuple)) and len(box1) == 4:
            x1_min, y1_min, x1_max, y1_max = float(box1[0]), float(box1[1]), float(box1[2]), float(box1[3])
        else:
            return 0.0
            
        if isinstance(box2, (list, tuple)) and len(box2) == 4:
            x2_min, y2_min, x2_max, y2_max = float(box2[0]), float(box2[1]), float(box2[2]), float(box2[3])
        else:
            return 0.0
    except (TypeError, ValueError):
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


def denormalize_box(norm_box, img_width, img_height):
    """Convert normalized coordinates [0,999] back to absolute coordinates"""
    x0 = (norm_box[0] / 999.0) * img_width
    y0 = (norm_box[1] / 999.0) * img_height
    x1 = (norm_box[2] / 999.0) * img_width
    y1 = (norm_box[3] / 999.0) * img_height
    return [x0, y0, x1, y1]


def cluster_boxes_by_iou(boxes, iou_threshold=0.5):
    """
    Cluster similar boxes using greedy IoU-based clustering
    
    Args:
        boxes: List of bounding boxes [[x0,y0,x1,y1], ...]
        iou_threshold: IoU threshold to consider boxes as similar
    
    Returns:
        List of clusters, each cluster is a list of box indices
    """
    if len(boxes) == 0:
        return []
    
    clusters = []
    box_to_cluster = {}  # box_idx -> cluster_idx
    
    for i, box in enumerate(boxes):
        matched_cluster = None
        best_iou = 0
        
        # Find best matching cluster
        for cluster_idx, cluster in enumerate(clusters):
            # Check IoU with all boxes in this cluster
            for j in cluster:
                iou = compute_iou(box, boxes[j])
                if iou >= iou_threshold and iou > best_iou:
                    best_iou = iou
                    matched_cluster = cluster_idx
        
        # Add to matched cluster or create new one
        if matched_cluster is not None:
            clusters[matched_cluster].append(i)
            box_to_cluster[i] = matched_cluster
        else:
            clusters.append([i])
            box_to_cluster[i] = len(clusters) - 1
    
    return clusters


def ensemble_predictions_for_batch(
    images,
    categories_list,
    model,
    n_samples=5,
    iou_threshold=0.5,
    vote_threshold=0.4,
    temperature=1.0,
):
    """
    Generate multiple predictions and ensemble them for a batch of samples
    
    Args:
        images: List of PIL Images
        categories_list: List of category lists for each image
        model: RexOmniWrapper instance
        n_samples: Number of predictions to generate per image
        iou_threshold: IoU threshold for clustering similar boxes
        vote_threshold: Minimum vote ratio to keep a box (0-1)
        temperature: Sampling temperature
    
    Returns:
        batch_ensemble_results: List of ensemble results per image
        batch_stats: List of statistics per image
    """
    batch_size = len(images)
    
    # Step 1: Generate n predictions for the entire batch
    all_batch_predictions = []
    
    # For vLLM backend, need to update sampling_params directly
    if hasattr(model, 'sampling_params'):
        # Save original sampling params
        original_temp = model.sampling_params.temperature
        original_top_p = model.sampling_params.top_p
    
    for sample_idx in range(n_samples):
        # ⭐ For vLLM: directly update SamplingParams
        if hasattr(model, 'sampling_params'):
            from vllm import SamplingParams
            # ⭐ Use different seed for each sample to ensure diversity
            random_seed = random.randint(0, 2**32 - 1)
            model.sampling_params = SamplingParams(
                max_tokens=model.max_tokens,
                temperature=temperature,  # ⭐ Use the provided temperature
                top_p=0.95,  # Higher top_p for more diversity
                seed=random_seed,  # ⭐ Random seed for diversity
                repetition_penalty=model.repetition_penalty,
                skip_special_tokens=model.skip_special_tokens,
                stop=model.stop,
            )
        else:
            # For transformers backend, modify temperature directly
            model.temperature = temperature
        
        # Batch inference for all images
        results = model.inference(
            images=images, 
            task="detection", 
            categories=categories_list,
        )
        
        all_batch_predictions.append(results)
    
    # Restore original sampling params
    if hasattr(model, 'sampling_params'):
        from vllm import SamplingParams
        model.sampling_params = SamplingParams(
            max_tokens=model.max_tokens,
            temperature=original_temp,
            top_p=original_top_p,
            repetition_penalty=model.repetition_penalty,
            skip_special_tokens=model.skip_special_tokens,
            stop=model.stop,
        )
    
    # Step 2: Process each image in the batch
    batch_ensemble_results = []
    batch_stats = []
    
    for img_idx in range(batch_size):
        categories = categories_list[img_idx]
        
        # Collect all predictions for this image across n_samples
        img_predictions = []
        for sample_idx in range(n_samples):
            img_predictions.append(all_batch_predictions[sample_idx][img_idx]["extracted_predictions"])
        
        # Ensemble for this image
        ensemble_results, stats = _ensemble_single_image(
            img_predictions,
            categories,
            n_samples,
            iou_threshold,
            vote_threshold
        )
        
        batch_ensemble_results.append(ensemble_results)
        batch_stats.append(stats)
    
    return batch_ensemble_results, batch_stats


def _ensemble_single_image(
    all_predictions,
    categories,
    n_samples,
    iou_threshold,
    vote_threshold
):
    """
    Ensemble predictions for a single image
    
    Args:
        all_predictions: List of prediction dicts from n_samples inferences
        categories: List of category names
        n_samples: Number of predictions
        iou_threshold: IoU threshold for clustering
        vote_threshold: Minimum vote ratio to keep a box
    
    Returns:
        ensemble_results: Dict[category -> List[dict with coords and confidence]]
        stats: Statistics about the ensemble process
    """
    ensemble_results = {}
    category_stats = {}
    
    for category in categories:
        # Collect all boxes for this category
        all_boxes = []
        for pred in all_predictions:
            if category in pred:
                for box_pred in pred[category]:
                    coords = box_pred['coords']
                    if isinstance(coords, (list, tuple)) and len(coords) == 4:
                        try:
                            box = [float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3])]
                            all_boxes.append(box)
                        except (TypeError, ValueError):
                            continue
        
        if len(all_boxes) == 0:
            ensemble_results[category] = []
            category_stats[category] = {
                'total_boxes': 0,
                'clusters': 0,
                'filtered_boxes': 0,
            }
            continue
        
        # Cluster similar boxes
        clusters = cluster_boxes_by_iou(all_boxes, iou_threshold)
        
        # Vote and filter
        filtered_boxes = []
        for cluster in clusters:
            vote_count = len(cluster)
            vote_ratio = vote_count / n_samples
            
            if vote_ratio >= vote_threshold:
                # Compute mean box as the final prediction
                cluster_boxes = [all_boxes[i] for i in cluster]
                mean_box = np.mean(cluster_boxes, axis=0)
                
                filtered_boxes.append({
                    'coords': mean_box.tolist(),
                    'confidence': float(vote_ratio),  # ⭐ Vote ratio as confidence
                    'vote_count': vote_count,
                    'cluster_size': len(cluster),
                })
        
        # Sort by confidence (descending)
        filtered_boxes.sort(key=lambda x: x['confidence'], reverse=True)
        
        ensemble_results[category] = filtered_boxes
        category_stats[category] = {
            'total_boxes': len(all_boxes),
            'clusters': len(clusters),
            'filtered_boxes': len(filtered_boxes),
            'filter_rate': len(filtered_boxes) / len(clusters) if len(clusters) > 0 else 0,
        }
    
    stats = {
        'n_samples': n_samples,
        'categories': category_stats,
        'total_boxes_before': sum(s['total_boxes'] for s in category_stats.values()),
        'total_boxes_after': sum(s['filtered_boxes'] for s in category_stats.values()),
    }
    
    return ensemble_results, stats


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
    
    Args:
        all_predictions: Dict[sample_id -> List[dict with coords and confidence]]
        all_ground_truths: Dict[sample_id -> List[boxes]]
        iou_threshold: IoU threshold
    
    Returns:
        ap: Average Precision
    """
    # Collect all predictions and ground truths
    all_pred_boxes = []
    all_gt_boxes = []
    
    for sample_id in all_ground_truths.keys():
        gt_boxes = all_ground_truths[sample_id]
        pred_data = all_predictions.get(sample_id, [])
        
        # Add match flag to each GT
        for gt_box in gt_boxes:
            all_gt_boxes.append({
                'sample_id': sample_id,
                'box': gt_box,
                'matched': False
            })
        
        # Add predictions with their confidence
        for pred in pred_data:
            all_pred_boxes.append({
                'sample_id': sample_id,
                'box': pred['coords'],
                'confidence': pred.get('confidence', 1.0),  # Use vote ratio as confidence
            })
    
    if len(all_gt_boxes) == 0:
        return 0.0
    
    if len(all_pred_boxes) == 0:
        return 0.0
    
    # Sort predictions by confidence (descending)
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
    parser = argparse.ArgumentParser(
        description="Evaluate model on OV-IGOD test set - Per-Affordance AP with Ensemble"
    )
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
        "--n_samples",
        type=int,
        default=5,
        help="Number of predictions to generate per image (ensemble size)"
    )
    parser.add_argument(
        "--vote_threshold",
        type=float,
        default=0.4,
        help="Minimum vote ratio to keep a box (0-1). E.g., 0.4 means box must appear in 40%% of samples"
    )
    parser.add_argument(
        "--iou_threshold",
        type=float,
        default=0.5,
        help="IoU threshold for clustering similar boxes"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (use 1.0 to match GRPO training distribution)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="evaluation_per_affordance_ensemble_results.json",
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
        help="Batch size for ensemble inference (note: total inferences = batch_size * n_samples)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of worker processes for data loading"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("OV-IGOD Test Set Evaluation - Per-Affordance AP with Ensemble")
    print("="*80)
    print(f"\nEnsemble Configuration:")
    print(f"  n_samples:       {args.n_samples}")
    print(f"  vote_threshold:  {args.vote_threshold} ({args.vote_threshold*100:.0f}%)")
    print(f"  iou_threshold:   {args.iou_threshold}")
    print(f"  temperature:     {args.temperature}")
    
    # Load test set
    print(f"\nLoading test set: {args.test_json}")
    with open(args.test_json, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    # Sample test_ids uniformly
    all_test_ids = list(test_data.keys())
    if args.max_samples and args.max_samples < len(all_test_ids):
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
        temperature=0.0,  # Will be overridden during ensemble
        top_p=0.95,
        repetition_penalty=1.05,
    )
    print("Model loaded successfully")
    
    # Create expanded samples (split by affordance)
    print("\nCreating per-affordance samples...")
    expanded_samples = []
    
    for img_id in test_ids:
        sample = test_data[img_id]
        image_path = os.path.join(args.image_root, f"{img_id}.jpg")
        
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue
        
        # Group bboxes by affordance
        affordance_bboxes = defaultdict(list)
        for bbox_item in sample['bboxes']:
            affordance = bbox_item['affordance']
            target_bboxes = parse_target_bboxes(bbox_item['target'])
            affordance_bboxes[affordance].extend(target_bboxes)
        
        # Create one sample per affordance
        for affordance, target_bboxes in affordance_bboxes.items():
            expanded_samples.append({
                'sample_id': f"{img_id}_{affordance}",
                'img_id': img_id,
                'image_path': image_path,
                'affordance': affordance,
                'target_bboxes': target_bboxes  # Normalized coordinates
            })
    
    print(f"Expanded {len(test_ids)} images into {len(expanded_samples)} per-affordance samples")
    
    # Collect predictions and ground truths per affordance
    per_affordance_predictions = defaultdict(dict)  # affordance -> {sample_id -> [boxes with confidence]}
    per_affordance_ground_truths = defaultdict(dict)  # affordance -> {sample_id -> [boxes]}
    all_ensemble_stats = []
    
    print(f"\nStarting ensemble inference...")
    print(f"Note: Each batch will be inferred {args.n_samples} times")
    print(f"Batch size: {args.batch_size}, Num workers: {args.num_workers}")
    print(f"Total inferences per batch: {args.batch_size * args.n_samples}")
    
    # Process in batches
    batch_size = args.batch_size
    num_batches = (len(expanded_samples) + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="Ensemble progress"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(expanded_samples))
        batch_samples = expanded_samples[start_idx:end_idx]
        
        # Prepare batch data
        batch_images = []
        batch_categories = []
        batch_metadata = []
        
        for sample_info in batch_samples:
            try:
                image = Image.open(sample_info['image_path']).convert("RGB")
                img_width, img_height = image.size
                
                # Convert GT to absolute coordinates
                abs_boxes = []
                for box in sample_info['target_bboxes']:
                    denorm_box = denormalize_box(box, img_width, img_height)
                    abs_boxes.append([float(denorm_box[0]), float(denorm_box[1]), 
                                     float(denorm_box[2]), float(denorm_box[3])])
                
                batch_images.append(image)
                batch_categories.append([sample_info['affordance']])  # ⭐ Only one affordance
                batch_metadata.append({
                    'sample_id': sample_info['sample_id'],
                    'affordance': sample_info['affordance'],
                    'gt_boxes': abs_boxes,
                    'img_width': img_width,
                    'img_height': img_height
                })
            except Exception as e:
                print(f"\nError loading {sample_info['image_path']}: {e}")
                continue
        
        # Batch ensemble inference
        if len(batch_images) > 0:
            try:
                # ⭐ Ensemble inference for the entire batch
                batch_ensemble_results, batch_stats = ensemble_predictions_for_batch(
                    images=batch_images,
                    categories_list=batch_categories,
                    model=rex,
                    n_samples=args.n_samples,
                    iou_threshold=args.iou_threshold,
                    vote_threshold=args.vote_threshold,
                    temperature=args.temperature,
                )
                
                # Process batch results
                for ensemble_results, stats, metadata in zip(batch_ensemble_results, batch_stats, batch_metadata):
                    sample_id = metadata['sample_id']
                    affordance = metadata['affordance']
                    
                    # Collect predicted boxes for this affordance
                    pred_boxes = []
                    for pred in ensemble_results.get(affordance, []):
                        pred_boxes.append({
                            'coords': pred['coords'],
                            'confidence': pred['confidence'],
                            'vote_count': pred['vote_count'],
                        })
                    
                    # Store per affordance
                    per_affordance_predictions[affordance][sample_id] = pred_boxes
                    per_affordance_ground_truths[affordance][sample_id] = metadata['gt_boxes']
                    all_ensemble_stats.append(stats)
                    
            except Exception as e:
                print(f"\nError processing batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Compute AP metrics per affordance
    print("\nComputing AP metrics...")
    
    all_affordances = sorted(per_affordance_ground_truths.keys())
    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    
    affordance_aps = {}  # affordance -> {AP@50, AP@75, AP@50:95}
    
    for affordance in all_affordances:
        preds = per_affordance_predictions[affordance]
        gts = per_affordance_ground_truths[affordance]
        
        # Count statistics
        total_gts = sum(len(boxes) for boxes in gts.values())
        total_preds = sum(len(boxes) for boxes in preds.values())
        
        # AP@50
        ap50 = evaluate_at_iou_threshold(preds, gts, 0.5)
        
        # AP@75
        ap75 = evaluate_at_iou_threshold(preds, gts, 0.75)
        
        # AP@50:95
        aps = []
        for iou_thresh in iou_thresholds:
            ap = evaluate_at_iou_threshold(preds, gts, iou_thresh)
            aps.append(ap)
        
        ap_50_95 = np.mean(aps)
        
        affordance_aps[affordance] = {
            'AP@50': float(ap50),
            'AP@75': float(ap75),
            'AP@50:95': float(ap_50_95),
            'detailed_aps': {f"AP@{iou:.2f}": float(ap) for iou, ap in zip(iou_thresholds, aps)},
            'num_gt_boxes': total_gts,
            'num_pred_boxes': total_preds,
            'num_samples': len(gts)
        }
    
    # Compute mAP (mean across all affordances)
    mAP_50 = np.mean([v['AP@50'] for v in affordance_aps.values()])
    mAP_75 = np.mean([v['AP@75'] for v in affordance_aps.values()])
    mAP_50_95 = np.mean([v['AP@50:95'] for v in affordance_aps.values()])
    
    # Compute ensemble statistics
    total_boxes_before = sum(s['total_boxes_before'] for s in all_ensemble_stats)
    total_boxes_after = sum(s['total_boxes_after'] for s in all_ensemble_stats)
    avg_boxes_before = total_boxes_before / len(all_ensemble_stats) if len(all_ensemble_stats) > 0 else 0
    avg_boxes_after = total_boxes_after / len(all_ensemble_stats) if len(all_ensemble_stats) > 0 else 0
    
    # Print results
    print("\n" + "="*80)
    print("Evaluation Results")
    print("="*80)
    
    total_samples = sum(stats['num_samples'] for stats in affordance_aps.values())
    total_gts = sum(stats['num_gt_boxes'] for stats in affordance_aps.values())
    total_preds = sum(stats['num_pred_boxes'] for stats in affordance_aps.values())
    
    print(f"\nDataset Statistics:")
    print(f"  Number of affordance samples: {total_samples}")
    print(f"  Number of affordances:        {len(all_affordances)}")
    print(f"  Total GT boxes:               {total_gts}")
    print(f"  Total predictions:            {total_preds}")
    
    print(f"\nEnsemble Statistics:")
    print(f"  Avg boxes before vote: {avg_boxes_before:.1f}")
    print(f"  Avg boxes after vote:  {avg_boxes_after:.1f}")
    print(f"  Filtering ratio:       {avg_boxes_after/avg_boxes_before*100:.1f}%")
    
    print(f"\nmAP Metrics:")
    print(f"  {'Metric':<15} {'Value':<10}")
    print(f"  {'-'*25}")
    print(f"  {'mAP@50':<15} {mAP_50:.4f} ({mAP_50*100:.2f}%)")
    print(f"  {'mAP@75':<15} {mAP_75:.4f} ({mAP_75*100:.2f}%)")
    print(f"  {'mAP@50:95':<15} {mAP_50_95:.4f} ({mAP_50_95*100:.2f}%)")
    
    # print(f"\nPer-Affordance AP Breakdown:")
    # print(f"{'Affordance':<80} {'AP@50':>10} {'AP@75':>10} {'AP@50:95':>10}")
    # print("-" * 112)
    
    # for affordance in sorted(affordance_aps.keys()):
    #     stats = affordance_aps[affordance]
    #     # Truncate affordance name if too long
    #     aff_name = affordance[:77] + "..." if len(affordance) > 80 else affordance
    #     print(f"{aff_name:<80} {stats['AP@50']:>10.4f} {stats['AP@75']:>10.4f} {stats['AP@50:95']:>10.4f}")
    
    # Save results
    results_dict = {
        "checkpoint": args.checkpoint,
        "evaluation_type": "per_affordance_ensemble",
        "ensemble_config": {
            "n_samples": args.n_samples,
            "vote_threshold": args.vote_threshold,
            "iou_threshold": args.iou_threshold,
            "temperature": args.temperature,
        },
        "num_images": len(test_ids),
        "num_expanded_samples": len(expanded_samples),
        "num_affordances": len(all_affordances),
        "affordances": list(all_affordances),
        "ensemble_stats": {
            "avg_boxes_before_vote": float(avg_boxes_before),
            "avg_boxes_after_vote": float(avg_boxes_after),
            "filtering_ratio": float(avg_boxes_after/avg_boxes_before) if avg_boxes_before > 0 else 0,
        },
        "per_affordance_metrics": affordance_aps,
        "mAP_metrics": {
            "mAP@50": float(mAP_50),
            "mAP@75": float(mAP_75),
            "mAP@50:95": float(mAP_50_95)
        }
    }
    
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)
    
    print(f"\nDetailed results saved to: {args.output_file}")
    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()

