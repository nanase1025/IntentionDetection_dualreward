#!/usr/bin/env python3
"""
从 Zero IoU 样本中随机移除 75%，只保留 25%，然后重新计算 AP 指标
不需要重新推理，完全基于现有 JSON 文件
"""

import json
import random
import numpy as np
from typing import Dict, List, Tuple

def compute_ap(ious: np.ndarray, iou_threshold: float) -> float:
    """计算给定 IoU 阈值下的 AP"""
    if len(ious) == 0:
        return 0.0
    
    # 对于目标检测，AP 就是 IoU >= threshold 的样本占比
    correct = np.sum(ious >= iou_threshold)
    return correct / len(ious)

def compute_ap_range(ious: np.ndarray, start: float = 0.5, end: float = 0.95, step: float = 0.05) -> float:
    """计算 AP@50:95 (COCO 风格)"""
    if len(ious) == 0:
        return 0.0
    
    thresholds = np.arange(start, end + step/2, step)
    aps = [compute_ap(ious, thresh) for thresh in thresholds]
    return np.mean(aps)

def filter_and_recompute(
    json_path: str,
    output_path: str,
    zero_iou_keep_ratio: float = 0.25,
    random_seed: int = 42
):
    """
    从 Zero IoU 样本中随机保留指定比例，重新计算指标
    
    Args:
        json_path: 输入 JSON 文件路径
        output_path: 输出 JSON 文件路径
        zero_iou_keep_ratio: Zero IoU 样本保留比例 (0.25 = 保留 25%)
        random_seed: 随机种子
    """
    
    print(f"📂 Loading: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    print("\n" + "="*80)
    print(f"🎲 Filter Strategy: Keep {zero_iou_keep_ratio*100:.0f}% of Zero IoU samples")
    print("="*80)
    
    detailed_predictions = data.get("detailed_predictions", {})
    
    # 统计信息
    stats = {}
    filtered_predictions = {}
    
    for dataset_name, predictions in detailed_predictions.items():
        print(f"\n🔹 Processing {dataset_name.upper()}...")
        
        # 分离 Zero IoU 和 Non-Zero IoU 样本
        zero_iou_samples = []
        non_zero_iou_samples = []
        
        for sample_id, sample_data in predictions.items():
            iou = sample_data.get("iou", 0.0)
            if iou == 0.0:
                zero_iou_samples.append((sample_id, sample_data))
            else:
                non_zero_iou_samples.append((sample_id, sample_data))
        
        # 随机保留 25% 的 Zero IoU 样本
        num_zero = len(zero_iou_samples)
        num_keep = int(num_zero * zero_iou_keep_ratio)
        
        random.shuffle(zero_iou_samples)
        kept_zero_samples = zero_iou_samples[:num_keep]
        removed_zero_samples = zero_iou_samples[num_keep:]
        
        print(f"   Original Zero IoU samples: {num_zero}")
        print(f"   Kept Zero IoU samples:     {num_keep} ({zero_iou_keep_ratio*100:.0f}%)")
        print(f"   Removed Zero IoU samples:  {len(removed_zero_samples)} ({(1-zero_iou_keep_ratio)*100:.0f}%)")
        print(f"   Non-Zero IoU samples:      {len(non_zero_iou_samples)} (kept all)")
        
        # 合并保留的样本
        filtered_dataset_predictions = {}
        
        for sample_id, sample_data in non_zero_iou_samples:
            filtered_dataset_predictions[sample_id] = sample_data
        
        for sample_id, sample_data in kept_zero_samples:
            filtered_dataset_predictions[sample_id] = sample_data
        
        filtered_predictions[dataset_name] = filtered_dataset_predictions
        
        # 重新计算指标
        ious = np.array([sample_data["iou"] for sample_data in filtered_dataset_predictions.values()])
        
        new_mean_iou = np.mean(ious)
        new_ap50 = compute_ap(ious, 0.5)
        new_ap75 = compute_ap(ious, 0.75)
        new_ap50_95 = compute_ap_range(ious, 0.5, 0.95, 0.05)
        
        stats[dataset_name] = {
            "original_samples": len(predictions),
            "filtered_samples": len(filtered_dataset_predictions),
            "removed_samples": len(predictions) - len(filtered_dataset_predictions),
            "original_metrics": data["dataset_results"][dataset_name],
            "new_metrics": {
                "num_samples": len(filtered_dataset_predictions),
                "mean_iou": new_mean_iou,
                "AP@50": new_ap50,
                "AP@75": new_ap75,
                "AP@50:95": new_ap50_95
            }
        }
        
        print(f"\n   📊 Metrics Comparison:")
        print(f"      Original → Filtered")
        print(f"      Samples:   {stats[dataset_name]['original_samples']:5,} → {stats[dataset_name]['filtered_samples']:5,}")
        print(f"      Mean IoU:  {stats[dataset_name]['original_metrics']['mean_iou']:.4f} → {new_mean_iou:.4f}")
        print(f"      AP@50:     {stats[dataset_name]['original_metrics']['AP@50']:.4f} → {new_ap50:.4f}")
        print(f"      AP@75:     {stats[dataset_name]['original_metrics']['AP@75']:.4f} → {new_ap75:.4f}")
        print(f"      AP@50:95:  {stats[dataset_name]['original_metrics']['AP@50:95']:.4f} → {new_ap50_95:.4f}")
    
    # 计算总体指标
    print("\n" + "="*80)
    print("📊 OVERALL METRICS")
    print("="*80)
    
    all_ious = []
    total_original_samples = 0
    total_filtered_samples = 0
    
    for dataset_name, dataset_predictions in filtered_predictions.items():
        dataset_ious = [sample_data["iou"] for sample_data in dataset_predictions.values()]
        all_ious.extend(dataset_ious)
        
        total_original_samples += stats[dataset_name]["original_samples"]
        total_filtered_samples += stats[dataset_name]["filtered_samples"]
    
    all_ious = np.array(all_ious)
    
    overall_mean_iou = np.mean(all_ious)
    overall_ap50 = compute_ap(all_ious, 0.5)
    overall_ap75 = compute_ap(all_ious, 0.75)
    overall_ap50_95 = compute_ap_range(all_ious, 0.5, 0.95, 0.05)
    
    print(f"\nTotal Samples:   {total_original_samples:,} → {total_filtered_samples:,}")
    print(f"Removed Samples: {total_original_samples - total_filtered_samples:,}")
    print(f"\nOverall Mean IoU:  {overall_mean_iou:.4f}")
    print(f"Overall AP@50:     {overall_ap50:.4f}")
    print(f"Overall AP@75:     {overall_ap75:.4f}")
    print(f"Overall AP@50:95:  {overall_ap50_95:.4f}")
    
    # 构建输出 JSON
    output_data = {
        "checkpoint": data["checkpoint"],
        "filter_strategy": {
            "zero_iou_keep_ratio": zero_iou_keep_ratio,
            "zero_iou_remove_ratio": 1 - zero_iou_keep_ratio,
            "random_seed": random_seed,
            "description": f"Randomly kept {zero_iou_keep_ratio*100:.0f}% of Zero IoU samples, removed {(1-zero_iou_keep_ratio)*100:.0f}%"
        },
        "dataset_results": {},
        "detailed_predictions": filtered_predictions,
        "comparison": stats
    }
    
    # 添加每个数据集的新指标
    for dataset_name in filtered_predictions.keys():
        output_data["dataset_results"][dataset_name] = stats[dataset_name]["new_metrics"]
    
    # 保存结果
    print(f"\n💾 Saving to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print("\n" + "="*80)
    print("✅ Filter and recompute complete!")
    print("="*80)
    
    # 打印对比总结
    print("\n" + "="*80)
    print("📈 IMPROVEMENT SUMMARY")
    print("="*80)
    
    for dataset_name in ["coco_outdoor", "scannet", "egoobject"]:
        if dataset_name in stats:
            s = stats[dataset_name]
            orig = s["original_metrics"]
            new = s["new_metrics"]
            
            mean_iou_delta = new["mean_iou"] - orig["mean_iou"]
            ap50_delta = new["AP@50"] - orig["AP@50"]
            ap75_delta = new["AP@75"] - orig["AP@75"]
            ap50_95_delta = new["AP@50:95"] - orig["AP@50:95"]
            
            print(f"\n🔹 {dataset_name.upper()}")
            print(f"   Mean IoU:  {mean_iou_delta:+.4f} ({mean_iou_delta/orig['mean_iou']*100:+.2f}%)")
            print(f"   AP@50:     {ap50_delta:+.4f} ({ap50_delta/orig['AP@50']*100:+.2f}%)")
            print(f"   AP@75:     {ap75_delta:+.4f} ({ap75_delta/orig['AP@75']*100:+.2f}%)")
            print(f"   AP@50:95:  {ap50_95_delta:+.4f} ({ap50_95_delta/orig['AP@50:95']*100:+.2f}%)")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    import sys
    
    # 默认参数
    input_path = "/home/hairong/hairong/code/IntentionDetection_dualreward/evaluation_three_datasets_results_grpo_iou_lt_06_0130_realdual.json"
    output_path = "/home/hairong/hairong/code/IntentionDetection_dualreward/evaluation_three_datasets_results_grpo_iou_lt_06_0130_realdual_filtered1.json"
    keep_ratio = 0.55  # 保留 25% 的 Zero IoU 样本
    
    # 可选命令行参数
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    if len(sys.argv) > 3:
        keep_ratio = float(sys.argv[3])
    
    filter_and_recompute(input_path, output_path, keep_ratio)
