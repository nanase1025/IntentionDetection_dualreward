#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Convert OV-IGOD dataset (JSON + images) to Rex-Omni TSV format for SFT/GRPO training.

OV-IGOD format:
    train.json / test.json:
    {
        "image_id": {
            "bboxes": [
                {"affordance": "...", "bbox": [x0,y0,x1,y1], "class_name": "...", ...},
                ...
            ]
        }
    }

Rex-Omni TSV format (per-affordance, matching GROUNDING_SINGLE_REGION_STAGE_XYXY):
    images.tsv:     <byte_offset>\t<base64_image>
    annotations.tsv: <img_byte_offset>\t{"boxes": [{"bbox": [x0,y0,x1,y1], "phrase": "affordance"}]}
    annotations.tsv.lineidx: byte offsets for each annotation line

Usage:
    python convert_ovigod_to_tsv.py \
        --input_json /home/hairong/hairong/code/ov-igod-dataset/train.json \
        --image_root /home/hairong/hairong/code/ov-igod-dataset/sunrgbd_jpgs \
        --output_dir /home/hairong/hairong/data/ov-igod-dataset \
        --split train

    python convert_ovigod_to_tsv.py \
        --input_json /home/hairong/hairong/code/ov-igod-dataset/test.json \
        --image_root /home/hairong/hairong/code/ov-igod-dataset/sunrgbd_jpgs \
        --output_dir /home/hairong/hairong/data/ov-igod-dataset \
        --split test
"""

import argparse
import base64
import io
import json
import os
from collections import defaultdict

from PIL import Image
from tqdm import tqdm


def load_ovigod_json(json_path):
    """Load OV-IGOD JSON and group bboxes by (image_id, affordance)."""
    with open(json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    samples = []
    for img_id, sample in raw_data.items():
        affordance_groups = defaultdict(list)
        for bbox_item in sample["bboxes"]:
            affordance = bbox_item["affordance"]
            bbox = bbox_item["bbox"]
            affordance_groups[affordance].append(bbox)

        for affordance, bboxes in affordance_groups.items():
            samples.append({
                "img_id": img_id,
                "image_name": f"{img_id}.jpg",
                "affordance": affordance,
                "bboxes": bboxes,
            })

    return samples


def image_to_base64(image_path):
    """Load image and convert to base64 JPEG string."""
    image = Image.open(image_path).convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def convert_to_tsv(samples, image_root, output_dir, split):
    """Convert samples to Rex-Omni TSV format."""
    os.makedirs(output_dir, exist_ok=True)

    img_tsv_path = os.path.join(output_dir, f"{split}.images.tsv")
    ann_tsv_path = os.path.join(output_dir, f"{split}.annotations.tsv")
    lineidx_path = os.path.join(output_dir, f"{split}.annotations.tsv.lineidx")

    img_cache = {}
    skipped = 0

    with open(img_tsv_path, "w") as f_img, \
         open(ann_tsv_path, "w") as f_ann, \
         open(lineidx_path, "w") as f_idx:

        img_offset = 0
        ann_offset = 0

        for sample in tqdm(samples, desc=f"Converting {split}"):
            image_path = os.path.join(image_root, sample["image_name"])
            if not os.path.exists(image_path):
                skipped += 1
                continue

            if sample["img_id"] not in img_cache:
                try:
                    img_b64 = image_to_base64(image_path)
                except Exception as e:
                    print(f"Warning: Failed to load {image_path}: {e}")
                    skipped += 1
                    continue
                img_line = f"{img_offset}\t{img_b64}\n"
                img_line_bytes = len(img_line.encode("utf-8"))
                f_img.write(img_line)
                img_cache[sample["img_id"]] = img_offset
                img_offset += img_line_bytes
            
            cached_img_offset = img_cache[sample["img_id"]]

            annotation = {
                "boxes": [
                    {"bbox": bbox, "phrase": sample["affordance"]}
                    for bbox in sample["bboxes"]
                ]
            }

            ann_line = f"{cached_img_offset}\t{json.dumps(annotation, ensure_ascii=False)}\n"
            ann_line_bytes = len(ann_line.encode("utf-8"))

            f_idx.write(f"{ann_offset}\n")
            f_ann.write(ann_line)
            ann_offset += ann_line_bytes

    return len(samples) - skipped, skipped


def main():
    parser = argparse.ArgumentParser(
        description="Convert OV-IGOD dataset to Rex-Omni TSV format"
    )
    parser.add_argument(
        "--input_json", type=str, required=True,
        help="Path to OV-IGOD train.json or test.json",
    )
    parser.add_argument(
        "--image_root", type=str, required=True,
        help="Path to sunrgbd_jpgs directory",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory for TSV files",
    )
    parser.add_argument(
        "--split", type=str, required=True, choices=["train", "test"],
        help="Dataset split (train or test)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print(f"Converting OV-IGOD {args.split} set to TSV format")
    print("=" * 60)

    print(f"\nLoading: {args.input_json}")
    samples = load_ovigod_json(args.input_json)
    print(f"Total (image, affordance) pairs: {len(samples)}")

    unique_images = len(set(s["img_id"] for s in samples))
    unique_affordances = len(set(s["affordance"] for s in samples))
    print(f"Unique images: {unique_images}")
    print(f"Unique affordance descriptions: {unique_affordances}")

    print(f"\nConverting to TSV...")
    converted, skipped = convert_to_tsv(
        samples, args.image_root, args.output_dir, args.split
    )

    print(f"\nDone!")
    print(f"  Converted: {converted} samples")
    print(f"  Skipped:   {skipped} samples (missing images)")
    print(f"\nOutput files:")
    print(f"  {os.path.join(args.output_dir, f'{args.split}.images.tsv')}")
    print(f"  {os.path.join(args.output_dir, f'{args.split}.annotations.tsv')}")
    print(f"  {os.path.join(args.output_dir, f'{args.split}.annotations.tsv.lineidx')}")


if __name__ == "__main__":
    main()
