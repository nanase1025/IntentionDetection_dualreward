#!/usr/bin/env python3
"""
Convert three intention datasets to Rex-Omni compatible TSV format
Fixed version - correct format with image_line_idx and annotation JSON
"""

import base64
import json
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from tqdm import tqdm


def load_image_from_data(image_data) -> Image.Image:
    """Load PIL Image from dataset image field"""
    if isinstance(image_data, dict):
        # HuggingFace datasets format
        if 'bytes' in image_data:
            img_bytes = image_data['bytes']
            # img_bytes is already bytes, not base64 encoded
            if isinstance(img_bytes, bytes):
                return Image.open(BytesIO(img_bytes)).convert('RGB')
            else:
                # If it's a string, try base64 decoding
                img_bytes = base64.b64decode(img_bytes)
                return Image.open(BytesIO(img_bytes)).convert('RGB')
        elif 'path' in image_data:
            return Image.open(image_data['path']).convert('RGB')
        else:
            raise ValueError(f"Unknown image dict format: {image_data.keys()}")
    elif isinstance(image_data, bytes):
        return Image.open(BytesIO(image_data)).convert('RGB')
    else:
        raise ValueError(f"Unknown image data type: {type(image_data)}")


def encode_image_to_base64(image: Image.Image) -> str:
    """Encode PIL Image to base64 string"""
    buffered = BytesIO()
    image.save(buffered, format="JPEG", quality=95)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def convert_bbox_xywh_to_xyxy(bbox: List[float]) -> List[float]:
    """Convert [x, y, w, h] to [x0, y0, x1, y1]"""
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def process_dataset(
    data_dir: str,
    output_dir: str,
    dataset_name: str,
    split: str
) -> int:
    """
    Process a single dataset split
    
    Args:
        data_dir: Path to dataset directory
        output_dir: Path to output directory
        dataset_name: Name of dataset (coco_outdoor, scannet, egoobject)
        split: train or test
    
    Returns:
        Number of samples processed
    """
    print(f"\n{'='*80}")
    print(f"Processing {dataset_name.upper()} - {split.upper()}")
    print(f"{'='*80}")
    
    data_path = Path(data_dir) / "data"
    
    # Read parquet files
    parquet_files = sorted(data_path.glob(f"{split}-*.parquet"))
    
    if not parquet_files:
        print(f"No parquet files found for {split}")
        return 0
    
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
    
    image_tsv_path = output_path / f"{dataset_name}_{split}.images.tsv"
    ann_tsv_path = output_path / f"{dataset_name}_{split}.annotations.tsv"
    ann_lineidx_path = output_path / f"{dataset_name}_{split}.annotations.tsv.lineidx"
    
    image_tsv = open(image_tsv_path, 'wb')  # binary mode for byte tracking
    ann_tsv = open(ann_tsv_path, 'wb')  # binary mode for byte tracking
    ann_lineidx = open(ann_lineidx_path, 'w', encoding='utf-8')
    
    num_processed = 0
    img_line_offset = 0
    ann_line_offset = 0
    
    for idx, row in tqdm(df_all.iterrows(), total=len(df_all), desc=f"{dataset_name}_{split}"):
        try:
            # Get image
            image = load_image_from_data(row['image'])
            img_width, img_height = image.size
            
            # Use only intention_1
            intention = row['intention_1']
            
            # Get bbox - handle different formats
            # NOTE: bbox can be a LIST of bboxes (multiple ground truths for same intention)
            bbox_raw = row['bbox']
            
            bboxes = []  # List to store all bboxes
            
            if isinstance(bbox_raw, str):
                # Parse string format: "[[x, y, w, h], [x, y, w, h], ...]"
                bbox_list = json.loads(bbox_raw)
                if bbox_list:
                    if isinstance(bbox_list[0], list):
                        # Multiple bboxes: [[x,y,w,h], [x,y,w,h], ...]
                        bboxes = bbox_list
                    else:
                        # Single bbox: [x, y, w, h]
                        bboxes = [bbox_list]
                else:
                    continue
            elif isinstance(bbox_raw, np.ndarray):
                # Handle numpy array
                if len(bbox_raw) > 0:
                    if bbox_raw.ndim == 2:
                        # 2D array: multiple bboxes
                        bboxes = bbox_raw.tolist()
                    elif isinstance(bbox_raw[0], np.ndarray):
                        # Array of arrays
                        bboxes = [b.tolist() for b in bbox_raw]
                    else:
                        # 1D array: single bbox
                        bboxes = [bbox_raw.tolist()]
                else:
                    continue
            elif isinstance(bbox_raw, list):
                if bbox_raw:
                    if isinstance(bbox_raw[0], list):
                        # Multiple bboxes: [[x,y,w,h], [x,y,w,h], ...]
                        bboxes = bbox_raw
                    else:
                        # Single bbox: [x, y, w, h]
                        bboxes = [bbox_raw]
                else:
                    continue
            else:
                continue
            
            # Convert all bboxes from [x, y, w, h] to [x0, y0, x1, y1]
            boxes_xyxy = []
            for bbox in bboxes:
                if len(bbox) != 4:
                    continue
                bbox_xyxy = convert_bbox_xywh_to_xyxy(bbox)
                boxes_xyxy.append(bbox_xyxy)
            
            if not boxes_xyxy:
                continue
            
            # Create annotation JSON in Rex-Omni format with ALL bboxes
            annotation = {
                "boxes": [
                    {
                        "bbox": bbox,
                        "phrase": intention
                    }
                    for bbox in boxes_xyxy
                ]
            }
            
            # Encode image to base64
            img_base64 = encode_image_to_base64(image)
            
            # Write image TSV line: {sample_id}\t{base64_image}
            sample_id = f"{dataset_name}_{row['id']}"
            image_line = f"{sample_id}\t{img_base64}\n"
            image_line_bytes = image_line.encode('utf-8')
            image_tsv.write(image_line_bytes)
            
            # Write annotation TSV line: {image_line_idx}\t{annotation_json}
            annotation_line = f"{img_line_offset}\t{json.dumps(annotation, ensure_ascii=False)}\n"
            annotation_line_bytes = annotation_line.encode('utf-8')
            ann_tsv.write(annotation_line_bytes)
            
            # Write annotation line index
            ann_lineidx.write(f"{ann_line_offset}\n")
            
            # Update offsets
            img_line_offset += len(image_line_bytes)
            ann_line_offset += len(annotation_line_bytes)
            
            num_processed += 1
        
        except Exception as e:
            print(f"\nError processing sample {idx} (id: {row.get('id', 'unknown')}): {e}")
            continue
    
    image_tsv.close()
    ann_tsv.close()
    ann_lineidx.close()
    
    print(f"✅ Processed {num_processed} samples")
    print(f"📁 Images TSV: {image_tsv_path}")
    print(f"📁 Annotations TSV: {ann_tsv_path}")
    print(f"📁 Line index: {ann_lineidx_path}")
    
    return num_processed


def main():
    """Main conversion function"""
    
    # Base directories
    data_base_dir = Path("/home/hairong/hairong/data")
    output_dir = Path("/home/hairong/hairong/data/intention_datasets_tsv_fixed")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    total_samples = 0
    
    # 1. COCO Outdoor - train and test
    print("\n" + "="*80)
    print("DATASET 1: COCO Outdoor Intention")
    print("="*80)
    
    coco_dir = data_base_dir / "coco-outdoor-intention"
    total_samples += process_dataset(str(coco_dir), str(output_dir), "coco_outdoor", "train")
    total_samples += process_dataset(str(coco_dir), str(output_dir), "coco_outdoor", "test")
    
    # 2. ScanNet - train and test
    print("\n" + "="*80)
    print("DATASET 2: ScanNet Intention")
    print("="*80)
    
    scannet_dir = data_base_dir / "scannet-intention"
    total_samples += process_dataset(str(scannet_dir), str(output_dir), "scannet", "train")
    total_samples += process_dataset(str(scannet_dir), str(output_dir), "scannet", "test")
    
    # 3. EgoObject - all as train
    print("\n" + "="*80)
    print("DATASET 3: EgoObject Intention")
    print("="*80)
    
    egoobject_dir = data_base_dir / "egoobject-intention"
    total_samples += process_dataset(str(egoobject_dir), str(output_dir), "egoobject", "train")
    
    # Summary
    print("\n" + "="*80)
    print("CONVERSION SUMMARY")
    print("="*80)
    print(f"Total samples processed: {total_samples}")
    print(f"Output directory: {output_dir}")
    print("\nGenerated files:")
    for tsv_file in sorted(output_dir.glob("*.tsv")):
        size_mb = tsv_file.stat().st_size / (1024 * 1024)
        print(f"  {tsv_file.name}: {size_mb:.1f} MB")
    
    print("\n✅ Conversion completed!")
    print("\nNext steps:")
    print("1. Update config: finetuning/configs/sft_intention_datasets.py")
    print("2. Run training: cd finetuning && bash scripts/sft_intention_datasets.sh")


if __name__ == "__main__":
    main()

