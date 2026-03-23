#!/usr/bin/env python3
"""
Analyze IoU distribution from evaluation results.
Shows how many samples are below/above a given IoU threshold.
"""

import json
import argparse


def analyze_iou_distribution(json_file, threshold=0.8):
    """Analyze IoU distribution from evaluation results."""
    
    # Load the evaluation results
    print(f"Loading evaluation results from: {json_file}")
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Count samples with IoU < threshold
    total_samples = 0
    iou_lt_threshold = 0
    iou_gte_threshold = 0
    no_prediction = 0
    
    # Check each dataset
    for dataset_name in ['coco_outdoor', 'scannet', 'egoobject']:
        if dataset_name in data['detailed_predictions']:
            predictions = data['detailed_predictions'][dataset_name]
            for sample_id, sample_data in predictions.items():
                total_samples += 1
                iou = sample_data.get('iou', 0.0)
                
                if iou < threshold:
                    iou_lt_threshold += 1
                else:
                    iou_gte_threshold += 1
                
                # Check if no prediction was made
                if 'predictions' in sample_data and len(sample_data['predictions']) == 0:
                    no_prediction += 1
    
    print(f"\n{'='*70}")
    print(f"IoU Distribution Analysis (Threshold: {threshold})")
    print(f"{'='*70}")
    print(f"Total samples: {total_samples:,}")
    print(f"")
    print(f"IoU < {threshold}:  {iou_lt_threshold:6,} ({iou_lt_threshold/total_samples*100:.2f}%)")
    print(f"IoU >= {threshold}: {iou_gte_threshold:6,} ({iou_gte_threshold/total_samples*100:.2f}%)")
    print(f"")
    print(f"No prediction: {no_prediction:,} ({no_prediction/total_samples*100:.2f}%)")
    print(f"{'='*70}")
    
    # Show IoU distribution by dataset
    print(f"\n{'='*70}")
    print(f"Per-Dataset Breakdown")
    print(f"{'='*70}")
    for dataset_name in ['coco_outdoor', 'scannet', 'egoobject']:
        if dataset_name in data['detailed_predictions']:
            predictions = data['detailed_predictions'][dataset_name]
            ds_total = len(predictions)
            ds_lt_threshold = sum(1 for s in predictions.values() if s.get('iou', 0.0) < threshold)
            ds_gte_threshold = ds_total - ds_lt_threshold
            
            print(f"\n{dataset_name.upper()}:")
            print(f"  Total: {ds_total:,}")
            print(f"  IoU < {threshold}:  {ds_lt_threshold:6,} ({ds_lt_threshold/ds_total*100:.2f}%)")
            print(f"  IoU >= {threshold}: {ds_gte_threshold:6,} ({ds_gte_threshold/ds_total*100:.2f}%)")
            
            # Show mean IoU from summary
            if dataset_name in data['dataset_results']:
                mean_iou = data['dataset_results'][dataset_name].get('mean_iou', 0.0)
                print(f"  Mean IoU: {mean_iou:.4f}")
    
    print(f"\n{'='*70}\n")
    
    return {
        'total': total_samples,
        'below_threshold': iou_lt_threshold,
        'above_threshold': iou_gte_threshold,
        'no_prediction': no_prediction,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Analyze IoU distribution from evaluation results'
    )
    parser.add_argument(
        'json_file',
        type=str,
        help='Path to evaluation results JSON file'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.8,
        help='IoU threshold for filtering (default: 0.8)'
    )
    
    args = parser.parse_args()
    
    analyze_iou_distribution(args.json_file, args.threshold)


if __name__ == '__main__':
    main()
