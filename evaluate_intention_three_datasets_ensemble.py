#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate fine-tuned Rex-Omni model on three intention detection datasets with Test-Time Ensemble

This script combines:
1. Three dataset evaluation (COCO Outdoor, ScanNet, EgoObject)
2. Temperature ensemble + voting for improved accuracy

Usage:
    python evaluate_intention_three_datasets_ensemble.py \
        --checkpoint finetuning/work_dirs/intention_grpo_filtered_epoch1/global_step_556/actor/huggingface \
        --n_samples 5 \
        --vote_threshold 0.4 \
        --temperature 1.0 \
        --max_samples 100
"""

import argparse
import json
import base64
import io
import random
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


def load_tsv_samples(img_tsv_file, ann_tsv_file, ann_lineidx_file):
    """Load samples from TSV files"""
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


def compute_ap(recalls, precisions):
    """
    Compute Average Precision (AP)
    Using all-point interpolation
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
    Evaluate predictions at a specific IoU threshold using one-to-one matching.
    
    For intention detection: each sample may have one or more GT boxes.
    We select the prediction with highest IoU against ANY GT box for each sample.
    AP = Accuracy = (number of samples with max IoU >= threshold) / total_samples
    
    Args:
        all_predictions: Dict[sample_id -> List[dict with coords and confidence]]
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
        pred_data = all_predictions.get(sample_id, [])
        
        if len(gt_boxes) == 0:
            continue
        
        # Find the best IoU across all GT boxes and all predictions
        best_iou = 0.0
        if len(pred_data) > 0:
            for pred in pred_data:
                pred_box = pred['coords']
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


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate model on three intention detection datasets with Test-Time Ensemble"
    )
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
        help="Directory containing COCO Outdoor test TSV files"
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
        default="evaluation_three_datasets_ensemble_results.json",
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
    
    args = parser.parse_args()
    
    print("="*80)
    print("Three Datasets Evaluation with Test-Time Ensemble")
    print("="*80)
    print(f"\nEnsemble Configuration:")
    print(f"  n_samples:       {args.n_samples}")
    print(f"  vote_threshold:  {args.vote_threshold} ({args.vote_threshold*100:.0f}%)")
    print(f"  iou_threshold:   {args.iou_threshold}")
    print(f"  temperature:     {args.temperature}")
    print(f"  batch_size:      {args.batch_size}")
    
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
    
    # Dataset configurations
    datasets = {
        'coco_outdoor': {
            'img_tsv': f"{args.coco_test_tsv}/coco_outdoor_test.images.tsv",
            'ann_tsv': f"{args.coco_test_tsv}/coco_outdoor_test.annotations.tsv",
            'ann_lineidx': f"{args.coco_test_tsv}/coco_outdoor_test.annotations.tsv.lineidx",
        },
        'scannet': {
            'img_tsv': f"{args.scannet_test_tsv}/scannet_test.images.tsv",
            'ann_tsv': f"{args.scannet_test_tsv}/scannet_test.annotations.tsv",
            'ann_lineidx': f"{args.scannet_test_tsv}/scannet_test.annotations.tsv.lineidx",
        },
        'egoobject': {
            'img_tsv': f"{args.egoobject_test_tsv}/egoobject_test.images.tsv",
            'ann_tsv': f"{args.egoobject_test_tsv}/egoobject_test.annotations.tsv",
            'ann_lineidx': f"{args.egoobject_test_tsv}/egoobject_test.annotations.tsv.lineidx",
        },
    }
    
    all_results = {}
    all_detailed_predictions = {}  # Store detailed predictions for all datasets
    
    # Evaluate each dataset
    for dataset_name, paths in datasets.items():
        print("\n" + "="*80)
        print(f"Evaluating {dataset_name.upper()} Test Set")
        print("="*80)
        
        # Load samples
        samples = load_tsv_samples(
            paths['img_tsv'],
            paths['ann_tsv'],
            paths['ann_lineidx']
        )
        
        # Subsample if needed
        if args.max_samples and args.max_samples < len(samples):
            step = len(samples) / args.max_samples
            samples = [samples[int(i * step)] for i in range(args.max_samples)]
        
        print(f"Total samples: {len(samples)}")
        print(f"Each sample will be inferred {args.n_samples} times")
        print(f"Total inferences per batch: {args.batch_size * args.n_samples}")
        
        # Collect predictions and ground truths
        predictions = {}
        ground_truths = {}
        all_ensemble_stats = []
        
        # Process in batches
        batch_size = args.batch_size
        num_batches = (len(samples) + batch_size - 1) // batch_size
        
        for batch_idx in tqdm(range(num_batches), desc=f"{dataset_name}"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(samples))
            batch_samples = samples[start_idx:end_idx]
            
            # Prepare batch data
            batch_images = []
            batch_categories = []
            batch_metadata = []
            
            for sample in batch_samples:
                image = sample['image']
                annotation = sample['annotation']
                
                # Extract phrase (category) and ground truth bbox(es)
                phrase = annotation['boxes'][0].get('phrase', 'object')
                gt_bboxes = [box['bbox'] for box in annotation['boxes']]  # Get all GT boxes
                
                batch_images.append(image)
                batch_categories.append([phrase])
                batch_metadata.append({
                    'sample_id': sample['sample_id'],
                    'phrase': phrase,
                    'gt_bboxes': [[float(b[0]), float(b[1]), float(b[2]), float(b[3])] for b in gt_bboxes],  # All GT boxes
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
                        sample_id = metadata['sample_id']
                        phrase = metadata['phrase']
                        
                        # Collect predicted boxes
                        pred_boxes = []
                        for pred in ensemble_results.get(phrase, []):
                            pred_boxes.append({
                                'coords': pred['coords'],
                                'confidence': pred['confidence'],
                                'vote_count': pred['vote_count'],
                            })
                        
                        # Store predictions and ground truths
                        predictions[sample_id] = pred_boxes
                        ground_truths[sample_id] = metadata['gt_bboxes']  # Store all GT boxes
                        all_ensemble_stats.append(stats)
                        
                except Exception as e:
                    print(f"\nError processing batch {batch_idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        # Compute AP metrics
        print("\nComputing AP metrics...")
        
        iou_thresholds = np.arange(0.5, 1.0, 0.05)
        
        # AP@50
        ap50 = evaluate_at_iou_threshold(predictions, ground_truths, 0.5)
        
        # AP@75
        ap75 = evaluate_at_iou_threshold(predictions, ground_truths, 0.75)
        
        # AP@50:95
        aps = []
        for iou_thresh in iou_thresholds:
            ap = evaluate_at_iou_threshold(predictions, ground_truths, iou_thresh)
            aps.append(ap)
        
        ap_50_95 = np.mean(aps)
        
        # Compute mean IoU
        all_ious = []
        sample_ious = {}  # Store IoU for each sample
        for sample_id in ground_truths.keys():
            gt_boxes = ground_truths[sample_id]
            pred_data = predictions.get(sample_id, [])
            
            if len(pred_data) > 0 and len(gt_boxes) > 0:
                # Take the best IoU for this sample
                best_iou = 0
                for pred in pred_data:
                    for gt_box in gt_boxes:
                        iou = compute_iou(pred['coords'], gt_box)
                        if iou > best_iou:
                            best_iou = iou
                all_ious.append(best_iou)
                sample_ious[sample_id] = float(best_iou)
            else:
                all_ious.append(0.0)
                sample_ious[sample_id] = 0.0
        
        mean_iou = np.mean(all_ious) if len(all_ious) > 0 else 0.0
        
        # Compute ensemble statistics
        total_boxes_before = sum(s['total_boxes_before'] for s in all_ensemble_stats)
        total_boxes_after = sum(s['total_boxes_after'] for s in all_ensemble_stats)
        avg_boxes_before = total_boxes_before / len(all_ensemble_stats) if len(all_ensemble_stats) > 0 else 0
        avg_boxes_after = total_boxes_after / len(all_ensemble_stats) if len(all_ensemble_stats) > 0 else 0
        
        # Store results
        all_results[dataset_name] = {
            'mean_iou': float(mean_iou),
            'AP@50': float(ap50),
            'AP@75': float(ap75),
            'AP@50:95': float(ap_50_95),
            'detailed_aps': {f"AP@{iou:.2f}": float(ap) for iou, ap in zip(iou_thresholds, aps)},
            'num_samples': len(samples),
            'num_gt_boxes': sum(len(boxes) for boxes in ground_truths.values()),
            'num_pred_boxes': sum(len(boxes) for boxes in predictions.values()),
            'ensemble_stats': {
                'avg_boxes_before_vote': float(avg_boxes_before),
                'avg_boxes_after_vote': float(avg_boxes_after),
                'filtering_ratio': float(avg_boxes_after/avg_boxes_before) if avg_boxes_before > 0 else 0,
            }
        }
        
        # Store detailed predictions for this dataset
        all_detailed_predictions[dataset_name] = {
            sample_id: {
                "ground_truth": ground_truths[sample_id],
                "predictions": [
                    {
                        "coords": pred['coords'],
                        "confidence": pred.get('confidence', 0.0),
                        "vote_count": pred.get('vote_count', 0)
                    }
                    for pred in predictions.get(sample_id, [])
                ],
                "iou": sample_ious[sample_id]
            }
            for sample_id in ground_truths.keys()
        }
        
        # Print results for this dataset
        print(f"\n{dataset_name.upper()} Results:")
        print(f"  Mean IoU:        {mean_iou*100:.2f}%")
        print(f"  AP@50:           {ap50*100:.2f}%")
        print(f"  AP@75:           {ap75*100:.2f}%")
        print(f"  AP@50:95:        {ap_50_95*100:.2f}%")
        print(f"  Ensemble Stats:")
        print(f"    Avg boxes before vote: {avg_boxes_before:.1f}")
        print(f"    Avg boxes after vote:  {avg_boxes_after:.1f}")
        print(f"    Filtering ratio:       {avg_boxes_after/avg_boxes_before*100:.1f}%")
    
    # Print summary
    print("\n" + "="*80)
    print("Summary - All Datasets")
    print("="*80)
    print(f"{'Dataset':<20} {'Mean IoU':>12} {'AP@50':>10} {'AP@75':>10} {'AP@50:95':>10}")
    print("-" * 64)
    
    for dataset_name in ['coco_outdoor', 'scannet', 'egoobject']:
        result = all_results[dataset_name]
        print(f"{dataset_name:<20} {result['mean_iou']*100:>11.2f}% {result['AP@50']*100:>9.2f}% "
              f"{result['AP@75']*100:>9.2f}% {result['AP@50:95']*100:>9.2f}%")
    
    # Save results
    results_dict = {
        "checkpoint": args.checkpoint,
        "evaluation_type": "three_datasets_ensemble",
        "ensemble_config": {
            "n_samples": args.n_samples,
            "vote_threshold": args.vote_threshold,
            "iou_threshold": args.iou_threshold,
            "temperature": args.temperature,
            "batch_size": args.batch_size,
        },
        "per_dataset_metrics": all_results,
        "detailed_predictions": all_detailed_predictions
    }
    
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)
    
    print(f"\nDetailed results saved to: {args.output_file}")
    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()

