#!/usr/bin/env python3
"""
Analyze IoU distribution from evaluation results
This helps you decide on appropriate thresholds for filtering
"""

import json
import sys
from pathlib import Path
import numpy as np


def analyze_iou_distribution(results_file: Path):
    """Analyze and display IoU distribution statistics"""
    with open(results_file) as f:
        data = json.load(f)
    
    dataset_name = data["dataset"]
    total = data["total_samples"]
    iou_scores = [r["iou"] for r in data["results"]]
    
    print(f"\n{'='*80}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*80}")
    print(f"Total samples: {total}")
    print(f"\nIoU Statistics:")
    print(f"  Mean:   {np.mean(iou_scores):.4f}")
    print(f"  Median: {np.median(iou_scores):.4f}")
    print(f"  Std:    {np.std(iou_scores):.4f}")
    print(f"  Min:    {np.min(iou_scores):.4f}")
    print(f"  Max:    {np.max(iou_scores):.4f}")
    
    print(f"\nPercentiles:")
    for p in [10, 25, 50, 75, 90]:
        print(f"  {p}th: {np.percentile(iou_scores, p):.4f}")
    
    print(f"\nDistribution by ranges:")
    ranges = [
        (0.0, 0.1, "Very Poor"),
        (0.1, 0.2, "Poor"),
        (0.2, 0.3, "Fair"),
        (0.3, 0.5, "Good"),
        (0.5, 0.7, "Very Good"),
        (0.7, 0.9, "Excellent"),
        (0.9, 1.0, "Perfect"),
    ]
    
    for min_val, max_val, label in ranges:
        count = sum(1 for iou in iou_scores if min_val <= iou < max_val)
        pct = count / total * 100
        bar = "█" * int(pct / 2)
        print(f"  [{min_val:.1f}, {max_val:.1f}) {label:12s}: {count:5d} ({pct:5.1f}%) {bar}")
    
    # Perfect scores (IoU = 1.0)
    perfect = sum(1 for iou in iou_scores if iou >= 1.0)
    if perfect > 0:
        pct = perfect / total * 100
        bar = "█" * int(pct / 2)
        print(f"  [1.0, 1.0] Perfect    : {perfect:5d} ({pct:5.1f}%) {bar}")
    
    print(f"\nRecommended thresholds for GRPO:")
    # Calculate thresholds based on distribution
    p25 = np.percentile(iou_scores, 25)
    p75 = np.percentile(iou_scores, 75)
    mean = np.mean(iou_scores)
    
    print(f"  Conservative (keep 50%): [{p25:.2f}, {p75:.2f}]")
    print(f"  Moderate (keep ~60-70%): [{max(0.15, p25-0.1):.2f}, {min(0.85, p75+0.1):.2f}]")
    print(f"  Relaxed (keep ~80%): [{max(0.1, p25-0.2):.2f}, {min(0.9, p75+0.2):.2f}]")
    
    return iou_scores


def main():
    output_dir = Path("work_dirs/filtered_data")
    
    if not output_dir.exists():
        print(f"Error: {output_dir} does not exist")
        print("Please run evaluation first: bash scripts/run_filter_data.sh")
        return
    
    datasets = ["coco_outdoor_train", "scannet_train", "egoobject_train"]
    
    all_iou_scores = []
    
    for dataset_name in datasets:
        results_file = output_dir / f"{dataset_name}_iou_scores.json"
        if not results_file.exists():
            print(f"Warning: {results_file} not found, skipping...")
            continue
        
        iou_scores = analyze_iou_distribution(results_file)
        all_iou_scores.extend(iou_scores)
    
    if len(all_iou_scores) > 0:
        print(f"\n{'='*80}")
        print(f"COMBINED STATISTICS (All Datasets)")
        print(f"{'='*80}")
        print(f"Total samples: {len(all_iou_scores)}")
        print(f"\nIoU Statistics:")
        print(f"  Mean:   {np.mean(all_iou_scores):.4f}")
        print(f"  Median: {np.median(all_iou_scores):.4f}")
        print(f"  Std:    {np.std(all_iou_scores):.4f}")
        
        print(f"\nRecommended filtering strategies:")
        print(f"\n1. Focus on 'learnable' samples (0.2-0.7):")
        count = sum(1 for iou in all_iou_scores if 0.2 <= iou <= 0.7)
        print(f"   bash scripts/apply_filter.sh 0.2 0.7")
        print(f"   → Keeps {count} samples ({count/len(all_iou_scores)*100:.1f}%)")
        
        print(f"\n2. More conservative (0.3-0.6):")
        count = sum(1 for iou in all_iou_scores if 0.3 <= iou <= 0.6)
        print(f"   bash scripts/apply_filter.sh 0.3 0.6")
        print(f"   → Keeps {count} samples ({count/len(all_iou_scores)*100:.1f}%)")
        
        print(f"\n3. More relaxed (0.15-0.75):")
        count = sum(1 for iou in all_iou_scores if 0.15 <= iou <= 0.75)
        print(f"   bash scripts/apply_filter.sh 0.15 0.75")
        print(f"   → Keeps {count} samples ({count/len(all_iou_scores)*100:.1f}%)")
        
        print(f"\n{'='*80}")


if __name__ == "__main__":
    main()

