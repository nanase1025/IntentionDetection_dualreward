#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
自定义测试脚本：可以自定义检测的意图描述
"""

from PIL import Image
from rex_omni import RexOmniWrapper, RexOmniVisualize

# 加载微调后的模型
print("Loading model...")
rex = RexOmniWrapper(
    model_path="finetuning/work_dirs/ovigod_sft-1000/checkpoint-93",
    backend="transformers",
    max_tokens=2048,
    temperature=0.0,
    top_p=0.05,
    top_k=1,
    repetition_penalty=1.05,
)

# 测试图片
image_path = "/workspace/hairong/data/ov-igod-dataset/sunrgbd_jpgs/1.jpg"
image = Image.open(image_path).convert("RGB")

print(f"\n📷 测试图片: {image_path}")
print(f"图片尺寸: {image.size}")

# 自定义检测意图（可以修改这里）
categories = [
    "I long for a comfortable place to rest and rejuvenate after a long day",
    "I need a convenient spot for my phone and a glass of water while I sleep",
    "I require soft lighting to create a calming atmosphere for reading before bed",
    # 你可以添加更多意图...
]

print(f"\n🎯 检测意图:")
for i, cat in enumerate(categories, 1):
    print(f"  {i}. {cat}")

print("\n⏳ 开始推理...")

# 推理
results = rex.inference(images=image, task="detection", categories=categories)
result = results[0]

# 打印结果
print("\n" + "="*80)
print("📊 检测结果")
print("="*80)

total_detections = 0
for category, predictions in result["extracted_predictions"].items():
    if predictions:
        print(f"\n✅ {category}")
        print(f"   检测到 {len(predictions)} 个目标:")
        for i, pred in enumerate(predictions, 1):
            coords = pred['coords']
            print(f"   [{i}] 坐标: [{coords[0]:.0f}, {coords[1]:.0f}, {coords[2]:.0f}, {coords[3]:.0f}]")
        total_detections += len(predictions)
    else:
        print(f"\n❌ {category}")
        print(f"   未检测到")

print(f"\n📈 总计检测到 {total_detections} 个目标")

# 可视化
output_path = "custom_test_result.jpg"
vis = RexOmniVisualize(
    image=image,
    predictions=result["extracted_predictions"],
    font_size=12,
    draw_width=3,
    show_labels=True,
)
vis.save(output_path)

print(f"\n💾 可视化结果已保存到: {output_path}")
print("\n✨ 测试完成！")

