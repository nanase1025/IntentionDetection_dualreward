#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate fine-tuned Rex-Omni model on OV-IGOD test set with Temperature Ensemble

This script implements self-ensemble strategy:
1. Generate n predictions with temperature > 0 (like training)
2. Cluster similar boxes using IoU
3. Vote: keep boxes that appear frequently (confidence = vote_ratio)

Usage:
    python evaluate_ovigod_ap_ensemble.py \
        --checkpoint finetuning/work_dirs/ovigod_sft/checkpoint-627 \
        --n_samples 5 \
        --vote_threshold 0.4 \
        --max_samples 100 \
        --batch_size 2
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
                        all_boxes.append(coords)
        
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


def ensemble_predictions_for_sample(
    image,
    categories,
    model,
    n_samples=5,
    iou_threshold=0.5,
    vote_threshold=0.4,
    temperature=1.0,
):
    """
    Generate multiple predictions and ensemble them for a single sample
    (Wrapper for backward compatibility - calls batch version with single image)
    
    Args:
        image: PIL Image
        categories: List of category names to detect
        model: RexOmniWrapper instance
        n_samples: Number of predictions to generate
        iou_threshold: IoU threshold for clustering similar boxes
        vote_threshold: Minimum vote ratio to keep a box (0-1)
        temperature: Sampling temperature
    
    Returns:
        ensemble_results: Dict[category -> List[dict with coords and confidence]]
        stats: Statistics about the ensemble process
    """
    batch_results, batch_stats = ensemble_predictions_for_batch(
        images=[image],
        categories_list=[categories],
        model=model,
        n_samples=n_samples,
        iou_threshold=iou_threshold,
        vote_threshold=vote_threshold,
        temperature=temperature,
    )
    return batch_results[0], batch_stats[0]


def _legacy_ensemble_predictions_for_sample(
    image,
    categories,
    model,
    n_samples=5,
    iou_threshold=0.5,
    vote_threshold=0.4,
    temperature=1.0,
):
    """
    Generate multiple predictions and ensemble them for a single sample (Legacy version)
    
    Args:
        image: PIL Image
        categories: List of category names to detect
        model: RexOmniWrapper instance
        n_samples: Number of predictions to generate
        iou_threshold: IoU threshold for clustering similar boxes
        vote_threshold: Minimum vote ratio to keep a box (0-1)
        temperature: Sampling temperature
    
    Returns:
        ensemble_results: Dict[category -> List[dict with coords and confidence]]
        stats: Statistics about the ensemble process
    """
    # Step 1: Generate n predictions
    all_predictions = []
    
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
        
        results = model.inference(
            images=image, 
            task="detection", 
            categories=categories,
        )
        
        all_predictions.append(results[0]["extracted_predictions"])
    
    # Restore original sampling params
    if hasattr(model, 'sampling_params'):
        model.sampling_params = SamplingParams(
            max_tokens=model.max_tokens,
            temperature=original_temp,
            top_p=original_top_p,
            repetition_penalty=model.repetition_penalty,
            skip_special_tokens=model.skip_special_tokens,
            stop=model.stop,
        )
    
    # Step 2: Cluster and vote for each category
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
                        all_boxes.append(coords)
        
        if len(all_boxes) == 0:
            ensemble_results[category] = []
            category_stats[category] = {
                'total_boxes': 0,
                'clusters': 0,
                'filtered_boxes': 0,
            }
            continue
        
        # Step 3: Cluster similar boxes
        clusters = cluster_boxes_by_iou(all_boxes, iou_threshold)
        
        # Step 4: Vote and filter
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
    """Compute Average Precision using VOC-style all-point interpolation"""
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
    """Evaluate all predictions at a specific IoU threshold"""
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
                'confidence': pred['confidence'],  # ⭐ Now we have real confidence!
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
        description="Evaluate model on OV-IGOD test set with Temperature Ensemble"
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
        default="evaluation_ensemble_results.json",
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
        help="Number of worker processes for data loading (use >1 for parallel processing)"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("OV-IGOD Test Set Evaluation - Temperature Ensemble + Voting")
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
        temperature=0.0,  # Will be overridden during ensemble
        top_p=0.95,
        repetition_penalty=1.05,
    )
    print("Model loaded successfully")
    
    # Collect all predictions and ground truths
    all_predictions = {}  # sample_id -> [{'coords': [...], 'confidence': ...}]
    all_ground_truths = {}  # sample_id -> [[x0,y0,x1,y1], ...]
    all_stats = {}  # sample_id -> stats
    
    print(f"\nStarting ensemble inference...")
    print(f"Note: Each batch will be inferred {args.n_samples} times")
    print(f"Batch size: {args.batch_size}, Num workers: {args.num_workers}")
    print(f"Total inferences per batch: {args.batch_size * args.n_samples}")
    
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
    
    for batch_idx in tqdm(range(num_batches), desc="Ensemble progress"):
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
                    img_id = metadata['img_id']
                    categories = batch_categories[batch_metadata.index(metadata)]
                    
                    # Collect all predicted boxes across categories
                    pred_boxes = []
                    for category in categories:
                        for pred in ensemble_results.get(category, []):
                            pred_boxes.append({
                                'coords': pred['coords'],
                                'confidence': pred['confidence'],
                                'vote_count': pred['vote_count'],
                            })
                    
                    all_predictions[img_id] = pred_boxes
                    all_stats[img_id] = stats
                    
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
    
    # Compute ensemble statistics
    total_boxes_before = sum(s['total_boxes_before'] for s in all_stats.values())
    total_boxes_after = sum(s['total_boxes_after'] for s in all_stats.values())
    avg_boxes_before = total_boxes_before / len(all_stats) if len(all_stats) > 0 else 0
    avg_boxes_after = total_boxes_after / len(all_stats) if len(all_stats) > 0 else 0
    
    # Print results
    print("\n" + "="*80)
    print("Evaluation Results")
    print("="*80)
    
    print(f"\nDataset Statistics:")
    print(f"  Number of samples:     {len(all_ground_truths)}")
    print(f"  Total GT boxes:        {sum(len(boxes) for boxes in all_ground_truths.values())}")
    
    print(f"\nEnsemble Statistics:")
    print(f"  Avg boxes before vote: {avg_boxes_before:.1f}")
    print(f"  Avg boxes after vote:  {avg_boxes_after:.1f}")
    print(f"  Filtering ratio:       {avg_boxes_after/avg_boxes_before*100:.1f}%")
    
    print(f"\nAP Metrics:")
    print(f"  {'Metric':<15} {'Value':<10}")
    print(f"  {'-'*25}")
    print(f"  {'AP@50':<15} {ap50:.4f} ({ap50*100:.2f}%)")
    print(f"  {'AP@75':<15} {ap75:.4f} ({ap75*100:.2f}%)")
    print(f"  {'AP@50:95':<15} {ap_50_95:.4f} ({ap_50_95*100:.2f}%)")
    
    # Save results
    results_dict = {
        "checkpoint": args.checkpoint,
        "ensemble_config": {
            "n_samples": args.n_samples,
            "vote_threshold": args.vote_threshold,
            "iou_threshold": args.iou_threshold,
            "temperature": args.temperature,
        },
        "test_samples": len(all_ground_truths),
        "total_gt": sum(len(boxes) for boxes in all_ground_truths.values()),
        "ensemble_stats": {
            "avg_boxes_before_vote": float(avg_boxes_before),
            "avg_boxes_after_vote": float(avg_boxes_after),
            "filtering_ratio": float(avg_boxes_after/avg_boxes_before) if avg_boxes_before > 0 else 0,
        },
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

