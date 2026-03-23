#!/usr/bin/env python3
"""
可视化 SFT 训练数据的 bounding box
验证 bbox 格式是否为 xyxy
"""

import os
import json
import base64
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import textwrap

# 数据路径
DATA_DIR = "/home/hairong/hairong/data/intention_datasets_tsv_fixed"
OUTPUT_DIR = "/home/hairong/hairong/code/IntentionDetection_dualreward/finetuning/ft_data_vis"

# 三个数据集
DATASETS = {
    "coco_outdoor": {
        "img_tsv": f"{DATA_DIR}/coco_outdoor_train.images.tsv",
        "ann_tsv": f"{DATA_DIR}/coco_outdoor_train.annotations.tsv",
    },
    "scannet": {
        "img_tsv": f"{DATA_DIR}/scannet_train.images.tsv",
        "ann_tsv": f"{DATA_DIR}/scannet_train.annotations.tsv",
    },
    "egoobject": {
        "img_tsv": f"{DATA_DIR}/egoobject_train.images.tsv",
        "ann_tsv": f"{DATA_DIR}/egoobject_train.annotations.tsv",
    },
}

# 颜色列表
COLORS = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow']


def load_tsv_line(tsv_file, line_idx):
    """读取 TSV 文件的指定行"""
    with open(tsv_file, 'r') as f:
        for i, line in enumerate(f):
            if i == line_idx:
                return line.strip()
    return None


def decode_image(img_data):
    """解码 base64 图片"""
    # img_data 格式: "idx\tbase64_string"
    parts = img_data.split('\t')
    if len(parts) >= 2:
        img_base64 = parts[1]
    else:
        img_base64 = parts[0]
    
    img_bytes = base64.b64decode(img_base64)
    img = Image.open(BytesIO(img_bytes))
    return img.convert('RGB')


def parse_annotation(ann_data):
    """解析标注数据"""
    # ann_data 格式: "idx\t{json}"
    parts = ann_data.split('\t', 1)
    if len(parts) >= 2:
        json_str = parts[1]
    else:
        json_str = parts[0]
    
    return json.loads(json_str)


def visualize_sample(img, annotation, output_path, dataset_name, sample_idx):
    """可视化单个样本"""
    # 创建绘图
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.imshow(img)
    
    boxes = annotation.get('boxes', [])
    
    # 用于存储所有 phrase 的列表
    phrases = []
    
    for i, box_info in enumerate(boxes):
        bbox = box_info['bbox']
        phrase = box_info['phrase']
        color = COLORS[i % len(COLORS)]
        
        # bbox 是 xyxy 格式: [x1, y1, x2, y2]
        x1, y1, x2, y2 = bbox
        
        # 绘制矩形框
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                             fill=False, edgecolor=color, linewidth=3)
        ax.add_patch(rect)
        
        # 在框上方添加编号
        ax.text(x1, y1 - 5, f"Box {i+1}", color=color, fontsize=12,
               fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 记录 phrase
        phrases.append(f"Box {i+1} ({color}): {phrase}")
    
    # 设置标题
    ax.set_title(f"{dataset_name} - Sample {sample_idx}\n"
                f"Image size: {img.width}x{img.height}, Boxes: {len(boxes)}\n"
                f"BBox format: xyxy (x1, y1, x2, y2)",
                fontsize=12)
    ax.axis('off')
    
    # 在图片下方添加 phrase 文本
    wrapped_phrases = []
    for p in phrases:
        wrapped = textwrap.fill(p, width=100)
        wrapped_phrases.append(wrapped)
    
    phrase_text = "\n\n".join(wrapped_phrases)
    
    # 调整布局，为文本留出空间
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.25)
    
    # 添加文本框
    fig.text(0.05, 0.02, phrase_text, fontsize=9, verticalalignment='bottom',
            wrap=True, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    # 保存图片
    plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.3)
    plt.close()
    
    print(f"Saved: {output_path}")
    
    # 打印 bbox 信息用于验证
    print(f"  BBox format verification:")
    for i, box_info in enumerate(boxes):
        bbox = box_info['bbox']
        x1, y1, x2, y2 = bbox
        print(f"    Box {i+1}: x1={x1:.2f}, y1={y1:.2f}, x2={x2:.2f}, y2={y2:.2f}")
        print(f"           Width={x2-x1:.2f}, Height={y2-y1:.2f}")
        if x2 > x1 and y2 > y1:
            print(f"           ✓ Valid xyxy format (x2>x1, y2>y1)")
        else:
            print(f"           ✗ WARNING: May not be xyxy format!")


def main():
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 每个数据集可视化前3个样本
    num_samples = 3
    
    for dataset_name, paths in DATASETS.items():
        print(f"\n{'='*60}")
        print(f"Processing: {dataset_name}")
        print(f"{'='*60}")
        
        img_tsv = paths['img_tsv']
        ann_tsv = paths['ann_tsv']
        
        for sample_idx in range(num_samples):
            print(f"\n--- Sample {sample_idx} ---")
            
            # 读取图片和标注
            img_data = load_tsv_line(img_tsv, sample_idx)
            ann_data = load_tsv_line(ann_tsv, sample_idx)
            
            if img_data is None or ann_data is None:
                print(f"  Failed to load sample {sample_idx}")
                continue
            
            # 解码图片
            try:
                img = decode_image(img_data)
            except Exception as e:
                print(f"  Failed to decode image: {e}")
                continue
            
            # 解析标注
            try:
                annotation = parse_annotation(ann_data)
            except Exception as e:
                print(f"  Failed to parse annotation: {e}")
                continue
            
            # 可视化并保存
            output_path = os.path.join(OUTPUT_DIR, f"{dataset_name}_sample_{sample_idx}.png")
            visualize_sample(img, annotation, output_path, dataset_name, sample_idx)
    
    print(f"\n{'='*60}")
    print(f"All visualizations saved to: {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
