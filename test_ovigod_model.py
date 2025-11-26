#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试脚本：用于验证在 OV-IGOD 数据集上微调后的 Rex-Omni 模型

使用方法：
    python test_ovigod_model.py --checkpoint work_dirs/ovigod_sft/checkpoint-1500
"""

import argparse
import os
from PIL import Image
import sys

# 添加 rex_omni 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from rex_omni import RexOmniWrapper, RexOmniVisualize


def main():
    parser = argparse.ArgumentParser(description="测试 OV-IGOD 微调后的模型")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="微调后的模型checkpoint路径，例如: finetuning/work_dirs/ovigod_sft/checkpoint-1500"
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default="/workspace/hairong/data/ov-igod-dataset/sunrgbd_jpgs/1.jpg",
        help="测试图片路径"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="ovigod_test_result.jpg",
        help="可视化结果保存路径"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="transformers",
        choices=["transformers", "vllm"],
        help="推理后端"
    )
    
    args = parser.parse_args()
    
    print(f"加载模型: {args.checkpoint}")
    
    # 加载微调后的模型
    rex = RexOmniWrapper(
        model_path=args.checkpoint,
        backend=args.backend,
        max_tokens=2048,
        temperature=0.0,
        top_p=0.05,
        top_k=1,
        repetition_penalty=1.05,
    )
    
    print(f"加载测试图片: {args.image_path}")
    
    # 测试图片
    image = Image.open(args.image_path).convert("RGB")
    
    # 使用意图描述作为检测目标（来自示例数据）
    categories = [
        "I long for a comfortable place to rest and rejuvenate after a long day",
        "I need a convenient spot for my phone and a glass of water while I sleep",
        "I want to keep my clothes organized and easily accessible for everyday use",
        "I require soft lighting to create a calming atmosphere for reading before bed"
    ]
    
    print(f"\n检测目标（意图描述）:")
    for i, cat in enumerate(categories, 1):
        print(f"  {i}. {cat}")
    
    print("\n开始推理...")
    
    # 推理
    results = rex.inference(images=image, task="detection", categories=categories)
    result = results[0]
    
    print("\n=== 原始输出 ===")
    print(result["raw_output"])
    
    print("\n=== 提取的预测结果 ===")
    for category, predictions in result["extracted_predictions"].items():
        print(f"\n{category}:")
        for pred in predictions:
            print(f"  - Type: {pred['type']}, Coords: {pred['coords']}")
    
    # 可视化
    print(f"\n生成可视化结果...")
    vis = RexOmniVisualize(
        image=image,
        predictions=result["extracted_predictions"],
        font_size=15,
        draw_width=3,
        show_labels=True,
    )
    vis.save(args.output_path)
    
    print(f"\n✅ 可视化结果已保存到: {args.output_path}")
    print(f"✅ 检测到 {sum(len(preds) for preds in result['extracted_predictions'].values())} 个目标")


if __name__ == "__main__":
    main()

