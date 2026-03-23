#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Convert three intention datasets to TSV format for Rex-Omni training

Datasets:
1. COCO Outdoor Intention (train + test)
2. ScanNet Intention (train + test)
3. EgoObject Intention (all as train)

Each sample uses only the FIRST intention (intention_1) for training.
"""

import argparse
import base64
import io
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


def convert_bbox_to_normalized(bbox: List[float], img_width: int, img_height: int) -> List[int]:
    """
    Convert bbox from various formats to normalized bins [0-999]
    
    Args:
        bbox: Can be [x, y, w, h] or [x0, y0, x1, y1]
        img_width: Image width
        img_height: Image height
    
    Returns:
        [x0_bin, y0_bin, x1_bin, y1_bin] in range [0, 999]
    """
    if len(bbox) == 4:
        # Assume COCO format: [x, y, w, h]
        if bbox[2] < img_width and bbox[3] < img_height:
            # Likely [x, y, w, h] format
            x0, y0, w, h = bbox
            x1 = x0 + w
            y1 = y0 + h
        else:
            # Likely [x0, y0, x1, y1] format already
            x0, y0, x1, y1 = bbox
    else:
        raise ValueError(f"Unexpected bbox format: {bbox}")
    
    # Normalize to [0, 1]
    x0_norm = max(0.0, min(1.0, x0 / img_width))
    y0_norm = max(0.0, min(1.0, y0 / img_height))
    x1_norm = max(0.0, min(1.0, x1 / img_width))
    y1_norm = max(0.0, min(1.0, y1 / img_height))
    
    # Convert to bins [0, 999]
    x0_bin = int(x0_norm * 999)
    y0_bin = int(y0_norm * 999)
    x1_bin = int(x1_norm * 999)
    y1_bin = int(y1_norm * 999)
    
    return [x0_bin, y0_bin, x1_bin, y1_bin]


def load_image_from_data(image_data) -> Image.Image:
    """Load PIL Image from various formats (dict, PIL Image, etc.)"""
    if isinstance(image_data, Image.Image):
        return image_data
    elif isinstance(image_data, dict):
        # Huggingface datasets format
        if 'bytes' in image_data:
            return Image.open(io.BytesIO(image_data['bytes']))
        elif 'path' in image_data:
            return Image.open(image_data['path'])
    raise ValueError(f"Unsupported image format: {type(image_data)}")


def encode_image_to_base64(image: Image.Image) -> str:
    """Encode PIL Image to base64 string"""
    buffered = io.BytesIO()
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image.save(buffered, format="JPEG", quality=95)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str


def create_target_string(bboxes: List[List[int]]) -> str:
    """
    Create target string in Rex-Omni format
    
    Args:
        bboxes: List of normalized bboxes [[x0, y0, x1, y1], ...]
    
    Returns:
        Target string like "<loc_100><loc_200><loc_300><loc_400>"
    """
    target_tokens = []
    for bbox in bboxes:
        for coord in bbox:
            target_tokens.append(f"<loc_{coord}>")
    return "".join(target_tokens)


def process_coco_outdoor(
    data_dir: str,
    output_dir: str,
    split: str = "train"
) -> Tuple[int, List[Dict]]:
    """
    Process COCO Outdoor Intention dataset
    
    Returns:
        (num_samples, samples_for_visualization)
    """
    print(f"\n{'='*80}")
    print(f"Processing COCO Outdoor - {split.upper()}")
    print(f"{'='*80}")
    
    data_path = Path(data_dir) / "data"
    
    # Read all parquet files for this split
    if split == "train":
        parquet_files = sorted(data_path.glob("train-*.parquet"))
    else:
        parquet_files = sorted(data_path.glob("test-*.parquet"))
    
    print(f"Found {len(parquet_files)} parquet files")
    
    all_samples = []
    for pf in parquet_files:
        df = pd.read_parquet(pf)
        all_samples.append(df)
    
    df_all = pd.concat(all_samples, ignore_index=True)
    print(f"Total samples: {len(df_all)}")
    
    # Prepare output
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    image_tsv_path = output_path / f"coco_outdoor_{split}.images.tsv"
    ann_tsv_path = output_path / f"coco_outdoor_{split}.annotations.tsv"
    
    image_tsv = open(image_tsv_path, 'w', encoding='utf-8')
    ann_tsv = open(ann_tsv_path, 'w', encoding='utf-8')
    
    vis_samples = []
    num_processed = 0
    
    for idx, row in tqdm(df_all.iterrows(), total=len(df_all), desc=f"COCO-Outdoor {split}"):
        try:
            # Get image
            image = load_image_from_data(row['image'])
            img_width, img_height = image.size
            sample_id = f"coco_outdoor_{row['id']}"
            
            # Use only intention_1
            intention = row['intention_1']
            
            # Get bbox (stored as string in JSON format)
            bbox_str = row['bbox']
            if isinstance(bbox_str, str):
                import ast
                bbox_list = ast.literal_eval(bbox_str)
            else:
                bbox_list = bbox_str
            
            if isinstance(bbox_list, list) and len(bbox_list) > 0:
                bbox = bbox_list[0]  # Take first bbox
            else:
                continue
            
            # Convert bbox to normalized bins
            norm_bbox = convert_bbox_to_normalized(bbox, img_width, img_height)
            target_string = create_target_string([norm_bbox])
            
            # Encode image
            img_base64 = encode_image_to_base64(image)
            
            # Create prompt (using intention as category)
            prompt = f"Detect {intention}. Output the bounding box coordinates in [x0, y0, x1, y1] format."
            
            # Write to TSV
            image_tsv.write(f"{sample_id}\t{img_base64}\n")
            ann_tsv.write(f"{sample_id}\t{prompt}\t{target_string}\n")
            
            num_processed += 1
            
            # Collect samples for visualization
            if num_processed <= 10:
                vis_samples.append({
                    'id': sample_id,
                    'image': image,
                    'intention': intention,
                    'bbox': bbox,
                    'target_category': row.get('target_category', 'unknown')
                })
        
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue
    
    image_tsv.close()
    ann_tsv.close()
    
    # Create line index
    create_line_index(ann_tsv_path)
    
    print(f"✅ Processed {num_processed} samples")
    print(f"📁 Images TSV: {image_tsv_path}")
    print(f"📁 Annotations TSV: {ann_tsv_path}")
    
    return num_processed, vis_samples


def process_scannet(
    data_dir: str,
    output_dir: str,
    split: str = "train"
) -> Tuple[int, List[Dict]]:
    """
    Process ScanNet Intention dataset
    
    Returns:
        (num_samples, samples_for_visualization)
    """
    print(f"\n{'='*80}")
    print(f"Processing ScanNet - {split.upper()}")
    print(f"{'='*80}")
    
    data_path = Path(data_dir) / "data"
    
    # Read all parquet files for this split
    if split == "train":
        parquet_files = sorted(data_path.glob("train-*.parquet"))
    else:
        parquet_files = sorted(data_path.glob("test-*.parquet"))
    
    print(f"Found {len(parquet_files)} parquet files")
    
    all_samples = []
    for pf in parquet_files:
        df = pd.read_parquet(pf)
        all_samples.append(df)
    
    df_all = pd.concat(all_samples, ignore_index=True)
    print(f"Total samples: {len(df_all)}")
    
    # Prepare output
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    image_tsv_path = output_path / f"scannet_{split}.images.tsv"
    ann_tsv_path = output_path / f"scannet_{split}.annotations.tsv"
    
    image_tsv = open(image_tsv_path, 'w', encoding='utf-8')
    ann_tsv = open(ann_tsv_path, 'w', encoding='utf-8')
    
    vis_samples = []
    num_processed = 0
    
    for idx, row in tqdm(df_all.iterrows(), total=len(df_all), desc=f"ScanNet {split}"):
        try:
            # Get image
            image = load_image_from_data(row['image'])
            img_width, img_height = image.size
            sample_id = f"scannet_{row['id']}"
            
            # Use only intention_1
            intention = row['intention_1']
            
            # Get bboxes (can be multiple)
            bbox_array = row['bbox']
            bboxes = []
            
            if isinstance(bbox_array, np.ndarray):
                # ScanNet format: numpy array of numpy arrays
                for bbox_item in bbox_array:
                    if isinstance(bbox_item, np.ndarray):
                        bboxes.append(bbox_item.tolist())
                    elif isinstance(bbox_item, list):
                        bboxes.append(bbox_item)
            elif isinstance(bbox_array, list):
                bboxes = bbox_array
            else:
                continue
            
            if len(bboxes) == 0:
                continue
            
            # Convert all bboxes to normalized bins
            norm_bboxes = []
            for bbox in bboxes:
                norm_bbox = convert_bbox_to_normalized(bbox, img_width, img_height)
                norm_bboxes.append(norm_bbox)
            
            target_string = create_target_string(norm_bboxes)
            
            # Encode image
            img_base64 = encode_image_to_base64(image)
            
            # Create prompt
            prompt = f"Detect {intention}. Output the bounding box coordinates in [x0, y0, x1, y1] format."
            
            # Write to TSV
            image_tsv.write(f"{sample_id}\t{img_base64}\n")
            ann_tsv.write(f"{sample_id}\t{prompt}\t{target_string}\n")
            
            num_processed += 1
            
            # Collect samples for visualization
            if num_processed <= 10:
                vis_samples.append({
                    'id': sample_id,
                    'image': image,
                    'intention': intention,
                    'bboxes': bboxes,
                    'target_category': row.get('target_category', 'unknown')
                })
        
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue
    
    image_tsv.close()
    ann_tsv.close()
    
    # Create line index
    create_line_index(ann_tsv_path)
    
    print(f"✅ Processed {num_processed} samples")
    print(f"📁 Images TSV: {image_tsv_path}")
    print(f"📁 Annotations TSV: {ann_tsv_path}")
    
    return num_processed, vis_samples


def process_egoobject(
    data_dir: str,
    output_dir: str
) -> Tuple[int, List[Dict]]:
    """
    Process EgoObject Intention dataset (all as train)
    
    Returns:
        (num_samples, samples_for_visualization)
    """
    print(f"\n{'='*80}")
    print(f"Processing EgoObject - TRAIN (all data)")
    print(f"{'='*80}")
    
    data_path = Path(data_dir) / "data"
    
    # Read all parquet files
    parquet_files = sorted(data_path.glob("train-*.parquet"))
    
    print(f"Found {len(parquet_files)} parquet files")
    
    all_samples = []
    for pf in parquet_files:
        df = pd.read_parquet(pf)
        all_samples.append(df)
    
    df_all = pd.concat(all_samples, ignore_index=True)
    print(f"Total samples: {len(df_all)}")
    
    # Prepare output
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    image_tsv_path = output_path / f"egoobject_train.images.tsv"
    ann_tsv_path = output_path / f"egoobject_train.annotations.tsv"
    
    image_tsv = open(image_tsv_path, 'w', encoding='utf-8')
    ann_tsv = open(ann_tsv_path, 'w', encoding='utf-8')
    
    vis_samples = []
    num_processed = 0
    
    for idx, row in tqdm(df_all.iterrows(), total=len(df_all), desc="EgoObject"):
        try:
            # Get image
            image = load_image_from_data(row['image'])
            img_width, img_height = image.size
            sample_id = f"egoobject_{row['id']}"
            
            # Use only intention_1
            intention = row['intention_1']
            
            # Get bbox
            bbox = row['bbox']
            if isinstance(bbox, np.ndarray):
                bbox = bbox.tolist()
            elif not isinstance(bbox, list):
                continue
            
            # Convert bbox to normalized bins
            norm_bbox = convert_bbox_to_normalized(bbox, img_width, img_height)
            target_string = create_target_string([norm_bbox])
            
            # Encode image
            img_base64 = encode_image_to_base64(image)
            
            # Create prompt
            prompt = f"Detect {intention}. Output the bounding box coordinates in [x0, y0, x1, y1] format."
            
            # Write to TSV
            image_tsv.write(f"{sample_id}\t{img_base64}\n")
            ann_tsv.write(f"{sample_id}\t{prompt}\t{target_string}\n")
            
            num_processed += 1
            
            # Collect samples for visualization
            if num_processed <= 10:
                vis_samples.append({
                    'id': sample_id,
                    'image': image,
                    'intention': intention,
                    'bbox': bbox,
                    'target_category': row.get('target_category', 'unknown')
                })
        
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue
    
    image_tsv.close()
    ann_tsv.close()
    
    # Create line index
    create_line_index(ann_tsv_path)
    
    print(f"✅ Processed {num_processed} samples")
    print(f"📁 Images TSV: {image_tsv_path}")
    print(f"📁 Annotations TSV: {ann_tsv_path}")
    
    return num_processed, vis_samples


def create_line_index(tsv_path: Path):
    """Create line index file for fast random access"""
    lineidx_path = Path(str(tsv_path) + ".lineidx")
    
    offsets = []
    with open(tsv_path, 'rb') as f:
        offsets.append(0)
        while f.readline():
            offsets.append(f.tell())
        offsets = offsets[:-1]  # Remove last offset (EOF)
    
    with open(lineidx_path, 'w', encoding='utf-8') as f:
        for offset in offsets:
            f.write(f"{offset}\n")
    
    print(f"📁 Line index: {lineidx_path}")


def visualize_samples(samples: List[Dict], output_dir: Path, dataset_name: str):
    """Visualize samples for verification"""
    vis_dir = output_dir / "visualizations" / dataset_name
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📊 Creating visualizations for {dataset_name}...")
    
    for sample in samples:
        try:
            image = sample['image'].copy()
            draw = ImageDraw.Draw(image)
            
            # Draw bbox(es)
            if 'bbox' in sample:
                # Single bbox
                bbox = sample['bbox']
                if len(bbox) == 4:
                    x, y, w, h = bbox
                    x1, y1 = x + w, y + h
                    draw.rectangle([x, y, x1, y1], outline='red', width=3)
            elif 'bboxes' in sample:
                # Multiple bboxes
                for bbox in sample['bboxes']:
                    if len(bbox) == 4:
                        x, y, w, h = bbox
                        x1, y1 = x + w, y + h
                        draw.rectangle([x, y, x1, y1], outline='red', width=3)
            
            # Add text
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
            except:
                font = ImageFont.load_default()
            
            # Wrap intention text
            intention = sample['intention']
            if len(intention) > 80:
                intention = intention[:77] + "..."
            
            draw.text((10, 10), f"Category: {sample['target_category']}", fill='yellow', font=font)
            draw.text((10, 30), f"Intention: {intention}", fill='yellow', font=font)
            
            # Save
            output_path = vis_dir / f"{sample['id']}.jpg"
            image.save(output_path)
        
        except Exception as e:
            print(f"Warning: Failed to visualize {sample['id']}: {e}")
    
    print(f"✅ Visualizations saved to: {vis_dir}")


def main():
    parser = argparse.ArgumentParser(description="Convert three intention datasets to TSV format")
    parser.add_argument(
        "--coco_dir",
        type=str,
        default="/home/hairong/hairong/data/coco-outdoor-intention",
        help="Path to COCO outdoor intention dataset"
    )
    parser.add_argument(
        "--scannet_dir",
        type=str,
        default="/home/hairong/hairong/data/scannet-intention",
        help="Path to ScanNet intention dataset"
    )
    parser.add_argument(
        "--egoobject_dir",
        type=str,
        default="/home/hairong/hairong/data/egoobject-intention",
        help="Path to EgoObject intention dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/hairong/hairong/data/intention_datasets_tsv",
        help="Output directory for TSV files"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Create visualization samples"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("Converting Intention Datasets to TSV Format")
    print("="*80)
    print(f"\nOutput directory: {args.output_dir}")
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    stats = {}
    
    # Process COCO Outdoor
    coco_train_count, coco_train_vis = process_coco_outdoor(args.coco_dir, args.output_dir, "train")
    coco_test_count, coco_test_vis = process_coco_outdoor(args.coco_dir, args.output_dir, "test")
    stats['coco_outdoor'] = {
        'train': coco_train_count,
        'test': coco_test_count,
        'total': coco_train_count + coco_test_count
    }
    
    # Process ScanNet
    scannet_train_count, scannet_train_vis = process_scannet(args.scannet_dir, args.output_dir, "train")
    scannet_test_count, scannet_test_vis = process_scannet(args.scannet_dir, args.output_dir, "test")
    stats['scannet'] = {
        'train': scannet_train_count,
        'test': scannet_test_count,
        'total': scannet_train_count + scannet_test_count
    }
    
    # Process EgoObject
    egoobject_train_count, egoobject_train_vis = process_egoobject(args.egoobject_dir, args.output_dir)
    stats['egoobject'] = {
        'train': egoobject_train_count,
        'test': 0,
        'total': egoobject_train_count
    }
    
    # Print summary
    print("\n" + "="*80)
    print("CONVERSION SUMMARY")
    print("="*80)
    
    total_train = 0
    total_test = 0
    
    for dataset, counts in stats.items():
        print(f"\n{dataset.upper()}:")
        print(f"  Train: {counts['train']:,}")
        print(f"  Test:  {counts['test']:,}")
        print(f"  Total: {counts['total']:,}")
        total_train += counts['train']
        total_test += counts['test']
    
    print(f"\nGRAND TOTAL:")
    print(f"  Train: {total_train:,}")
    print(f"  Test:  {total_test:,}")
    print(f"  Total: {total_train + total_test:,}")
    
    # Visualize samples
    if args.visualize:
        print("\n" + "="*80)
        print("Creating Visualizations")
        print("="*80)
        
        visualize_samples(coco_train_vis, output_path, "coco_outdoor_train")
        visualize_samples(coco_test_vis, output_path, "coco_outdoor_test")
        visualize_samples(scannet_train_vis, output_path, "scannet_train")
        visualize_samples(scannet_test_vis, output_path, "scannet_test")
        visualize_samples(egoobject_train_vis, output_path, "egoobject_train")
    
    # Save statistics
    stats_path = output_path / "conversion_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\n📊 Statistics saved to: {stats_path}")
    
    print("\n✅ All conversions completed successfully!")
    print(f"\n📁 Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()

