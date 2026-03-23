#!/usr/bin/env python3
"""
分析 evaluation JSON 文件中 IoU 为 0 的样本在不同子集中的分布
"""

import json
import sys

def analyze_zero_iou(json_path: str):
    """统计每个子集中 IoU 为 0 的样本数量"""
    
    print(f"📂 Loading: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    print("\n" + "="*80)
    print("📊 Zero IoU Analysis (IoU = 0.0)")
    print("="*80)
    
    # 统计每个子集
    detailed_predictions = data.get("detailed_predictions", {})
    
    total_zero_iou = 0
    total_samples = 0
    
    results = {}
    
    for dataset_name, predictions in detailed_predictions.items():
        zero_iou_count = 0
        num_samples = len(predictions)
        
        # 统计 IoU 为 0 的样本
        for sample_id, sample_data in predictions.items():
            iou = sample_data.get("iou", 0.0)
            if iou == 0.0:
                zero_iou_count += 1
        
        results[dataset_name] = {
            "zero_iou": zero_iou_count,
            "total": num_samples,
            "percentage": (zero_iou_count / num_samples * 100) if num_samples > 0 else 0
        }
        
        total_zero_iou += zero_iou_count
        total_samples += num_samples
    
    # 打印结果
    print("\n📋 Per-Dataset Statistics:")
    print("-" * 80)
    
    for dataset_name in ["coco_outdoor", "scannet", "egoobject"]:
        if dataset_name in results:
            r = results[dataset_name]
            print(f"\n🔹 {dataset_name.upper()}")
            print(f"   Zero IoU samples: {r['zero_iou']:,}")
            print(f"   Total samples:    {r['total']:,}")
            print(f"   Percentage:       {r['percentage']:.2f}%")
    
    print("\n" + "="*80)
    print(f"📊 OVERALL STATISTICS")
    print("="*80)
    print(f"Total Zero IoU samples: {total_zero_iou:,}")
    print(f"Total samples:          {total_samples:,}")
    print(f"Overall percentage:     {(total_zero_iou / total_samples * 100):.2f}%")
    print("="*80)
    
    # 额外分析：IoU 分布
    print("\n📈 IoU Distribution Analysis:")
    print("-" * 80)
    
    iou_ranges = {
        "0.0": (0.0, 0.0),
        "(0.0, 0.2)": (0.0, 0.2),
        "[0.2, 0.4)": (0.2, 0.4),
        "[0.4, 0.6)": (0.4, 0.6),
        "[0.6, 0.8)": (0.6, 0.8),
        "[0.8, 1.0]": (0.8, 1.0)
    }
    
    for dataset_name, predictions in detailed_predictions.items():
        print(f"\n🔹 {dataset_name.upper()}")
        
        distribution = {range_name: 0 for range_name in iou_ranges.keys()}
        
        for sample_id, sample_data in predictions.items():
            iou = sample_data.get("iou", 0.0)
            
            for range_name, (low, high) in iou_ranges.items():
                if range_name == "0.0":
                    if iou == 0.0:
                        distribution[range_name] += 1
                        break
                elif range_name == "(0.0, 0.2)":
                    if 0.0 < iou < 0.2:
                        distribution[range_name] += 1
                        break
                elif range_name == "[0.8, 1.0]":
                    if 0.8 <= iou <= 1.0:
                        distribution[range_name] += 1
                        break
                else:
                    if low <= iou < high:
                        distribution[range_name] += 1
                        break
        
        num_samples = len(predictions)
        for range_name, count in distribution.items():
            percentage = (count / num_samples * 100) if num_samples > 0 else 0
            print(f"   {range_name:15s}: {count:5,} ({percentage:5.2f}%)")
    
    print("\n" + "="*80)
    print("✅ Analysis complete!")
    print("="*80)

if __name__ == "__main__":
    json_path = "/home/hairong/hairong/code/IntentionDetection_dualreward/evaluation_three_datasets_results_grpo_iou_lt_08_0115.json"
    
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
    
    analyze_zero_iou(json_path)
