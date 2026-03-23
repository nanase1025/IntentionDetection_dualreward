#!/usr/bin/env python3
"""
Filter training data based on IoU scores from evaluation results.
This script reads the evaluation JSON (with IoU for each sample) and creates
filtered TSV datasets for GRPO training.

Usage:
    python3 filter_train_data_by_iou.py \
        --eval_results evaluation_three_datasets_train_results.json \
        --iou_threshold 0.3 \
        --output_dir finetuning/work_dirs/filtered_data_grpo
"""

import os
import json
import argparse
from typing import Dict, Set
import base64


def load_evaluation_results(eval_json_path: str) -> Dict[str, Dict[str, float]]:
    """
    Load evaluation results and extract IoU for each sample.
    
    Returns:
        Dict[dataset_name -> Dict[sample_id -> iou]]
    """
    print(f"📂 Loading evaluation results from {eval_json_path}")
    
    with open(eval_json_path, 'r') as f:
        data = json.load(f)
    
    iou_scores = {}
    for dataset_name, predictions in data['detailed_predictions'].items():
        iou_scores[dataset_name] = {
            sample_id: pred_data['iou']
            for sample_id, pred_data in predictions.items()
        }
        print(f"   {dataset_name}: {len(iou_scores[dataset_name])} samples")
    
    return iou_scores


def filter_samples_by_iou(
    iou_scores: Dict[str, float],
    iou_threshold: float
) -> Set[str]:
    """
    Filter samples based on IoU threshold.
    
    Returns:
        Set of sample_ids that pass the threshold
    """
    passed_samples = {
        sample_id
        for sample_id, iou in iou_scores.items()
        if iou >= iou_threshold
    }
    
    return passed_samples


def filter_tsv_dataset(
    image_tsv_path: str,
    anno_tsv_path: str,
    anno_lineidx_path: str,
    passed_samples: Set[str],
    output_image_tsv_path: str,
    output_anno_tsv_path: str,
    output_anno_lineidx_path: str
):
    """
    Filter TSV dataset files based on passed sample IDs.
    """
    print(f"  Reading from {image_tsv_path}")
    print(f"  Reading from {anno_tsv_path}")
    
    # Read original files
    with open(image_tsv_path, 'rb') as img_f, \
         open(anno_tsv_path, 'r', encoding='utf-8') as anno_f:
        
        # Read all annotations first
        annotations = []
        for line in anno_f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                byte_offset = int(parts[0])
                anno_json_str = parts[1]
                annotations.append({
                    'byte_offset': byte_offset,
                    'anno_json': anno_json_str
                })
        
        print(f"  Total samples: {len(annotations)}")
        
        # Filter samples
        filtered_data = []
        for anno_entry in annotations:
            # Read image data
            byte_offset = anno_entry['byte_offset']
            img_f.seek(byte_offset)
            img_line = img_f.readline().decode('utf-8').strip()
            
            # Extract sample_id
            img_parts = img_line.split('\t')
            if len(img_parts) < 2:
                continue
            
            sample_id = img_parts[0]
            
            # Check if sample passes threshold
            if sample_id in passed_samples:
                filtered_data.append({
                    'sample_id': sample_id,
                    'image_line': img_line,
                    'anno_json': anno_entry['anno_json']
                })
        
        print(f"  Filtered samples: {len(filtered_data)}")
        print(f"  Filtering ratio: {len(filtered_data)/len(annotations)*100:.2f}%")
    
    # Write filtered data
    os.makedirs(os.path.dirname(output_image_tsv_path), exist_ok=True)
    
    print(f"  Writing to {output_image_tsv_path}")
    print(f"  Writing to {output_anno_tsv_path}")
    print(f"  Writing to {output_anno_lineidx_path}")
    
    with open(output_image_tsv_path, 'wb') as img_out_f, \
         open(output_anno_tsv_path, 'w', encoding='utf-8') as anno_out_f, \
         open(output_anno_lineidx_path, 'w', encoding='utf-8') as lineidx_out_f:
        
        current_byte_offset = 0
        
        for entry in filtered_data:
            # Write image line
            img_line_bytes = (entry['image_line'] + '\n').encode('utf-8')
            img_out_f.write(img_line_bytes)
            
            # Write annotation line with updated byte offset
            anno_line = f"{current_byte_offset}\t{entry['anno_json']}\n"
            anno_out_f.write(anno_line)
            
            # Write line index (same as byte offset)
            lineidx_out_f.write(f"{current_byte_offset}\n")
            
            # Update byte offset
            current_byte_offset += len(img_line_bytes)
    
    print(f"  ✅ Filtered dataset saved")


def analyze_iou_distribution(iou_scores: Dict[str, float]) -> Dict:
    """
    Analyze IoU distribution and return statistics.
    """
    import numpy as np
    
    ious = list(iou_scores.values())
    
    stats = {
        'mean': float(np.mean(ious)),
        'median': float(np.median(ious)),
        'std': float(np.std(ious)),
        'min': float(np.min(ious)),
        'max': float(np.max(ious)),
        'percentiles': {
            '25': float(np.percentile(ious, 25)),
            '50': float(np.percentile(ious, 50)),
            '75': float(np.percentile(ious, 75)),
            '90': float(np.percentile(ious, 90)),
            '95': float(np.percentile(ious, 95)),
        },
        'thresholds': {}
    }
    
    # Calculate sample counts at different thresholds
    for threshold in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        count = sum(1 for iou in ious if iou >= threshold)
        ratio = count / len(ious) * 100
        stats['thresholds'][f'{threshold:.1f}'] = {
            'count': count,
            'ratio': f'{ratio:.2f}%'
        }
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Filter training data by IoU threshold')
    parser.add_argument('--eval_results', type=str, required=True,
                        help='Path to evaluation JSON file (with IoU for each sample)')
    parser.add_argument('--iou_threshold', type=float, default=0.3,
                        help='IoU threshold for filtering (default: 0.3)')
    parser.add_argument('--output_dir', type=str, 
                        default='finetuning/work_dirs/filtered_data_grpo',
                        help='Output directory for filtered datasets')
    parser.add_argument('--analyze_only', action='store_true',
                        help='Only analyze IoU distribution without filtering')
    
    args = parser.parse_args()
    
    print("="*80)
    print("FILTER TRAINING DATA BY IoU")
    print("="*80)
    print(f"Evaluation results: {args.eval_results}")
    print(f"IoU threshold: {args.iou_threshold}")
    print(f"Output directory: {args.output_dir}")
    print("="*80)
    
    # Load evaluation results
    iou_scores = load_evaluation_results(args.eval_results)
    
    # Analyze IoU distribution for each dataset
    print(f"\n{'='*80}")
    print("📊 IoU DISTRIBUTION ANALYSIS")
    print(f"{'='*80}")
    
    for dataset_name, dataset_ious in iou_scores.items():
        print(f"\n{dataset_name.upper()}:")
        stats = analyze_iou_distribution(dataset_ious)
        
        print(f"  Mean IoU:   {stats['mean']:.4f}")
        print(f"  Median IoU: {stats['median']:.4f}")
        print(f"  Std Dev:    {stats['std']:.4f}")
        print(f"  Range:      [{stats['min']:.4f}, {stats['max']:.4f}]")
        print(f"\n  Percentiles:")
        for p, val in stats['percentiles'].items():
            print(f"    {p}th: {val:.4f}")
        
        print(f"\n  Sample retention at different thresholds:")
        print(f"    {'Threshold':<12} {'Count':<10} {'Ratio'}")
        print(f"    {'-'*35}")
        for thresh, info in stats['thresholds'].items():
            print(f"    {thresh:<12} {info['count']:<10} {info['ratio']}")
    
    if args.analyze_only:
        print(f"\n{'='*80}")
        print("Analysis complete. Use without --analyze_only to filter data.")
        print("="*80)
        return
    
    # Dataset paths
    datasets_config = {
        'coco_outdoor': {
            'image_tsv': '/home/hairong/hairong/data/intention_datasets_tsv/coco_outdoor_train.images.tsv',
            'anno_tsv': '/home/hairong/hairong/data/intention_datasets_tsv/coco_outdoor_train.annotations.tsv',
            'anno_lineidx': '/home/hairong/hairong/data/intention_datasets_tsv/coco_outdoor_train.annotations.tsv.lineidx',
        },
        'scannet': {
            'image_tsv': '/home/hairong/hairong/data/intention_datasets_tsv/scannet_train.images.tsv',
            'anno_tsv': '/home/hairong/hairong/data/intention_datasets_tsv/scannet_train.annotations.tsv',
            'anno_lineidx': '/home/hairong/hairong/data/intention_datasets_tsv/scannet_train.annotations.tsv.lineidx',
        },
        'egoobject': {
            'image_tsv': '/home/hairong/hairong/data/intention_datasets_tsv/egoobject_train.images.tsv',
            'anno_tsv': '/home/hairong/hairong/data/intention_datasets_tsv/egoobject_train.annotations.tsv',
            'anno_lineidx': '/home/hairong/hairong/data/intention_datasets_tsv/egoobject_train.annotations.tsv.lineidx',
        }
    }
    
    # Filter each dataset
    print(f"\n{'='*80}")
    print(f"🔍 FILTERING WITH IoU >= {args.iou_threshold}")
    print(f"{'='*80}")
    
    for dataset_name, config in datasets_config.items():
        print(f"\n{dataset_name.upper()}:")
        
        # Get passed samples
        passed_samples = filter_samples_by_iou(
            iou_scores[dataset_name],
            args.iou_threshold
        )
        
        print(f"  Samples passing threshold: {len(passed_samples)}")
        
        # Output paths
        output_image_tsv = os.path.join(args.output_dir, dataset_name, 'train_filtered.images.tsv')
        output_anno_tsv = os.path.join(args.output_dir, dataset_name, 'train_filtered.annotations.tsv')
        output_anno_lineidx = os.path.join(args.output_dir, dataset_name, 'train_filtered.annotations.tsv.lineidx')
        
        # Filter dataset
        filter_tsv_dataset(
            config['image_tsv'],
            config['anno_tsv'],
            config['anno_lineidx'],
            passed_samples,
            output_image_tsv,
            output_anno_tsv,
            output_anno_lineidx
        )
    
    # Save filtering summary
    summary_path = os.path.join(args.output_dir, 'filtering_summary.json')
    summary = {
        'iou_threshold': args.iou_threshold,
        'datasets': {}
    }
    
    for dataset_name, dataset_ious in iou_scores.items():
        passed_samples = filter_samples_by_iou(dataset_ious, args.iou_threshold)
        total = len(dataset_ious)
        filtered = len(passed_samples)
        
        summary['datasets'][dataset_name] = {
            'total_samples': total,
            'filtered_samples': filtered,
            'filtering_ratio': f'{filtered/total*100:.2f}%',
            'mean_iou_original': float(sum(dataset_ious.values()) / total),
            'mean_iou_filtered': float(sum(iou for sid, iou in dataset_ious.items() if sid in passed_samples) / filtered) if filtered > 0 else 0.0
        }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*80}")
    print("✅ FILTERING COMPLETE")
    print(f"{'='*80}")
    print(f"Filtered datasets saved to: {args.output_dir}")
    print(f"Summary saved to: {summary_path}")
    print(f"\n{'Dataset':<20} {'Original':<10} {'Filtered':<10} {'Ratio'}")
    print("-"*55)
    for dataset_name, info in summary['datasets'].items():
        print(f"{dataset_name:<20} {info['total_samples']:<10} {info['filtered_samples']:<10} {info['filtering_ratio']}")
    print("="*80)


if __name__ == "__main__":
    main()

