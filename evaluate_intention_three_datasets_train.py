#!/usr/bin/env python3
"""
Evaluate Rex-Omni model on three intention detection TRAIN datasets
This script evaluates on COCO Outdoor, ScanNet, and EgoObject TRAIN sets
Output format is identical to test evaluation for GRPO data filtering
"""

import os
import sys
import json
import argparse
from PIL import Image
import io
import base64
from tqdm import tqdm
import numpy as np
from typing import List, Dict, Tuple, Optional

from rex_omni import RexOmniWrapper


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """
    Compute IoU between two boxes.
    Args:
        box1, box2: [x0, y0, x1, y1] in absolute coordinates
    Returns:
        IoU value
    """
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


def load_tsv_samples(
    image_tsv_path: str,
    anno_tsv_path: str,
    anno_lineidx_path: str,
    max_samples: Optional[int] = None
) -> List[Dict]:
    """
    Load samples from TSV files (standard training format).
    
    Standard TSV format (same as used in official training):
    - annotations.tsv: {image_byte_offset}\t{annotation_json}
    - images.tsv: standard TSV with base64 images
    - lineidx: byte offsets for annotations.tsv
    
    NOTE: The first column in annotations.tsv is the BYTE OFFSET in images.tsv,
    not a line number!
    
    Annotation JSON format:
    {
        "boxes": [
            {"bbox": [x0, y0, x1, y1], "phrase": "intention query"}
        ]
    }
    
    Args:
        image_tsv_path: Path to .images.tsv file
        anno_tsv_path: Path to .annotations.tsv file
        anno_lineidx_path: Path to .annotations.tsv.lineidx file
        max_samples: Maximum number of samples to load (None = all)
    
    Returns:
        List of dicts with keys: sample_id, image, annotation
    """
    samples = []
    
    # Read line index (byte offsets for annotations)
    print(f"  Loading line index from {anno_lineidx_path}")
    with open(anno_lineidx_path, 'r') as f:
        line_offsets = [int(line.strip()) for line in f]
    
    print(f"  Found {len(line_offsets)} annotations")
    
    # Limit samples if requested
    if max_samples:
        line_offsets = line_offsets[:max_samples]
        print(f"  Limited to {len(line_offsets)} samples")
    
    # First pass: find which image byte offsets we need
    print(f"  Finding required image byte offsets from annotations")
    required_image_offsets = set()
    with open(anno_tsv_path, 'rb') as anno_file:
        for byte_offset in line_offsets:
            try:
                anno_file.seek(byte_offset)
                line = anno_file.readline().decode('utf-8').strip()
                parts = line.split('\t')
                if len(parts) >= 2:
                    image_byte_offset = int(parts[0])
                    required_image_offsets.add(image_byte_offset)
            except Exception as e:
                print(f"  Warning: Error reading annotation at offset {byte_offset}: {e}")
                continue
    
    print(f"  Need to load {len(required_image_offsets)} unique images")
    
    # Load only required images using byte offsets
    print(f"  Loading required images from {image_tsv_path}")
    images_by_offset = {}
    with open(image_tsv_path, 'rb') as img_file:
        for img_offset in tqdm(required_image_offsets, desc="  Reading images"):
            try:
                img_file.seek(img_offset)
                line = img_file.readline().decode('utf-8').strip()
                parts = line.split('\t')
                if len(parts) >= 2:
                    image_base64 = parts[1]
                    images_by_offset[img_offset] = image_base64
            except Exception as e:
                print(f"  Warning: Error loading image at offset {img_offset}: {e}")
                continue
    
    print(f"  Loaded {len(images_by_offset)} images")
    
    # Load annotations and create samples
    print(f"  Loading annotations and creating samples")
    with open(anno_tsv_path, 'rb') as anno_file:
        for idx, byte_offset in enumerate(tqdm(line_offsets, desc="  Creating samples")):
            try:
                # Read annotation line
                anno_file.seek(byte_offset)
                line = anno_file.readline().decode('utf-8').strip()
                parts = line.split('\t')
                
                if len(parts) < 2:
                    print(f"  Warning: Skipping malformed annotation line at offset {byte_offset}")
                    continue
                
                # Parse standard format: {image_byte_offset}\t{annotation_json}
                image_byte_offset = int(parts[0])
                annotation_json = json.loads(parts[1])
                
                # Get image by byte offset
                if image_byte_offset not in images_by_offset:
                    print(f"  Warning: Image not loaded for byte offset {image_byte_offset}")
                    continue
                
                image_base64 = images_by_offset[image_byte_offset]
                image_bytes = base64.b64decode(image_base64)
                image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                
                # Annotation is already in correct format
                # Just need to add a sample_id for tracking
                sample_id = f"sample_{idx}"
                
                samples.append({
                    'sample_id': sample_id,
                    'image': image,
                    'annotation': annotation_json,
                    'query': annotation_json['boxes'][0]['phrase'] if annotation_json.get('boxes') else ''
                })
                
            except Exception as e:
                print(f"  Error loading sample {idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    print(f"  Successfully loaded {len(samples)} samples")
    return samples


def evaluate_at_iou_threshold(
    predictions: Dict[str, List[List[float]]],
    ground_truths: Dict[str, List[List[float]]],
    iou_threshold: float
) -> float:
    """
    Compute AP at a specific IoU threshold.
    For intention detection (one-to-one matching), AP = accuracy.
    """
    total = len(ground_truths)
    if total == 0:
        return 0.0
    
    tp = 0
    for sample_id in ground_truths.keys():
        gt_boxes = ground_truths[sample_id]
        pred_boxes = predictions.get(sample_id, [])
        
        if len(pred_boxes) == 0 or len(gt_boxes) == 0:
            continue
        
        # Find best IoU match
        best_iou = 0
        for pred_box in pred_boxes:
            for gt_box in gt_boxes:
                iou = compute_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
        
        # Count as TP if IoU >= threshold
        if best_iou >= iou_threshold:
            tp += 1
    
    return tp / total


def evaluate_dataset(
    samples: List[Dict],
    rex: RexOmniWrapper,
    dataset_name: str
) -> Tuple[Dict[str, List[List[float]]], Dict[str, List[List[float]]], Dict[str, float]]:
    """
    Evaluate model on a dataset.
    
    Returns:
        predictions: Dict[sample_id -> List of predicted boxes]
        ground_truths: Dict[sample_id -> List of GT boxes]
        sample_ious: Dict[sample_id -> float IoU]
    """
    predictions = {}
    ground_truths = {}
    sample_ious = {}
    
    for sample in tqdm(samples, desc=f"Evaluating {dataset_name}"):
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
            
            predictions[sample_id] = pred_boxes
            ground_truths[sample_id] = gt_boxes
            
            # Compute IoU for this sample
            if pred_boxes and gt_boxes:
                max_iou = 0
                for pred_box in pred_boxes:
                    for gt_box in gt_boxes:
                        iou = compute_iou(pred_box, gt_box)
                        if iou > max_iou:
                            max_iou = iou
                sample_ious[sample_id] = float(max_iou)
            else:
                sample_ious[sample_id] = 0.0
        except Exception as e:
            print(f"Error processing {sample_id}: {e}")
            predictions[sample_id] = []
            ground_truths[sample_id] = gt_boxes
            sample_ious[sample_id] = 0.0
    
    return predictions, ground_truths, sample_ious


def main():
    parser = argparse.ArgumentParser(description='Evaluate Rex-Omni on three intention detection TRAIN datasets')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output_file', type=str, default='evaluation_three_datasets_train_results.json',
                        help='Output JSON file path')
    parser.add_argument('--backend', type=str, default='vllm', choices=['transformers', 'vllm'],
                        help='Inference backend')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum samples per dataset (for testing)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("EVALUATION ON TRAIN SETS")
    print("="*80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Backend: {args.backend}")
    print(f"Output file: {args.output_file}")
    if args.max_samples:
        print(f"Max samples per dataset: {args.max_samples}")
    print("="*80)
    
    # Initialize model
    print("\n📦 Loading model...")
    rex = RexOmniWrapper(
        model_path=args.checkpoint,
        backend=args.backend,
        max_tokens=2048,
        temperature=0.0
    )
    print("✅ Model loaded successfully\n")
    
    # Dataset configurations (TRAIN sets)
    datasets_config = {
        'coco_outdoor': {
            'image_tsv': '/home/hairong/hairong/data/intention_datasets_tsv_fixed/coco_outdoor_train.images.tsv',
            'anno_tsv': '/home/hairong/hairong/data/intention_datasets_tsv_fixed/coco_outdoor_train.annotations.tsv',
            'anno_lineidx': '/home/hairong/hairong/data/intention_datasets_tsv_fixed/coco_outdoor_train.annotations.tsv.lineidx',
        },
        'scannet': {
            'image_tsv': '/home/hairong/hairong/data/intention_datasets_tsv_fixed/scannet_train.images.tsv',
            'anno_tsv': '/home/hairong/hairong/data/intention_datasets_tsv_fixed/scannet_train.annotations.tsv',
            'anno_lineidx': '/home/hairong/hairong/data/intention_datasets_tsv_fixed/scannet_train.annotations.tsv.lineidx',
        },
        'egoobject': {
            'image_tsv': '/home/hairong/hairong/data/intention_datasets_tsv_fixed/egoobject_train.images.tsv',
            'anno_tsv': '/home/hairong/hairong/data/intention_datasets_tsv_fixed/egoobject_train.annotations.tsv',
            'anno_lineidx': '/home/hairong/hairong/data/intention_datasets_tsv_fixed/egoobject_train.annotations.tsv.lineidx',
        }
    }
    
    # Evaluate each dataset
    all_results = {}
    all_detailed_predictions = {}
    
    for dataset_name, config in datasets_config.items():
        print(f"\n{'='*80}")
        print(f"📊 Evaluating {dataset_name.upper()} TRAIN set")
        print(f"{'='*80}")
        
        # Load samples
        print(f"\n1️⃣ Loading data...")
        samples = load_tsv_samples(
            config['image_tsv'],
            config['anno_tsv'],
            config['anno_lineidx'],
            max_samples=args.max_samples
        )
        
        # Run evaluation
        print(f"\n2️⃣ Running inference...")
        predictions, ground_truths, sample_ious = evaluate_dataset(samples, rex, dataset_name)
        
        # Compute metrics
        print(f"\n3️⃣ Computing metrics...")
        
        # Mean IoU
        mean_iou = np.mean(list(sample_ious.values())) if sample_ious else 0.0
        
        # AP at different thresholds
        ap50 = evaluate_at_iou_threshold(predictions, ground_truths, 0.50)
        ap75 = evaluate_at_iou_threshold(predictions, ground_truths, 0.75)
        
        # AP@50:95
        iou_thresholds = np.arange(0.50, 1.0, 0.05)
        aps = [evaluate_at_iou_threshold(predictions, ground_truths, t) for t in iou_thresholds]
        ap_50_95 = np.mean(aps)
        
        # Store results
        all_results[dataset_name] = {
            'num_samples': len(samples),
            'mean_iou': float(mean_iou),
            'AP@50': float(ap50),
            'AP@75': float(ap75),
            'AP@50:95': float(ap_50_95),
        }
        
        # Store detailed predictions
        all_detailed_predictions[dataset_name] = {
            sample_id: {
                "ground_truth": ground_truths[sample_id],
                "predictions": predictions[sample_id],
                "iou": sample_ious[sample_id]
            }
            for sample_id in predictions.keys()
        }
        
        # Print results
        print(f"\n📈 Results for {dataset_name}:")
        print(f"   Samples:    {all_results[dataset_name]['num_samples']}")
        print(f"   Mean IoU:   {all_results[dataset_name]['mean_iou']:.4f}")
        print(f"   AP@50:      {all_results[dataset_name]['AP@50']:.4f}")
        print(f"   AP@75:      {all_results[dataset_name]['AP@75']:.4f}")
        print(f"   AP@50:95:   {all_results[dataset_name]['AP@50:95']:.4f}")
    
    # Save results
    print(f"\n{'='*80}")
    print("💾 Saving results...")
    print(f"{'='*80}")
    
    output_data = {
        'checkpoint': args.checkpoint,
        'dataset_results': all_results,
        'detailed_predictions': all_detailed_predictions
    }
    
    with open(args.output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✅ Results saved to: {args.output_file}")
    
    # Print summary
    print(f"\n{'='*80}")
    print("📊 FINAL SUMMARY (TRAIN SETS)")
    print(f"{'='*80}")
    print(f"{'Dataset':<20} {'Samples':<10} {'Mean IoU':<12} {'AP@50':<10} {'AP@75':<10} {'AP@50:95':<10}")
    print("-"*80)
    for dataset_name, results in all_results.items():
        print(f"{dataset_name:<20} {results['num_samples']:<10} "
              f"{results['mean_iou']:<12.4f} {results['AP@50']:<10.4f} "
              f"{results['AP@75']:<10.4f} {results['AP@50:95']:<10.4f}")
    print("="*80)
    
    print(f"\n✅ Evaluation complete! Results saved to: {args.output_file}")
    print(f"📝 This file can be used for GRPO data filtering based on IoU scores.")


if __name__ == "__main__":
    main()

