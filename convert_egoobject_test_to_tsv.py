#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Convert EgoObject test set (egoobject-intention-eval) from Parquet to TSV format

This script handles the specific format of EgoObject test data and avoids previous pitfalls:
1. Image data is in dict format {'bytes': ...}, not base64
2. TSV format uses byte offsets, not line indices
3. Proper handling of intentions and bboxes
"""

import os
import base64
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import io


def load_image_from_data(image_data):
    """
    Load image from various data formats
    
    EgoObject format: image_data is a dict with 'bytes' key
    """
    if isinstance(image_data, dict) and 'bytes' in image_data:
        # EgoObject format: dict with 'bytes' key
        img_bytes = image_data['bytes']
        image = Image.open(io.BytesIO(img_bytes))
        return image
    elif isinstance(image_data, bytes):
        # Direct bytes
        image = Image.open(io.BytesIO(image_data))
        return image
    else:
        raise ValueError(f"Unsupported image data format: {type(image_data)}")


def convert_egoobject_test_to_tsv(
    parquet_dir,
    output_dir,
    dataset_name="egoobject_test"
):
    """
    Convert EgoObject test Parquet files to TSV format
    
    Args:
        parquet_dir: Directory containing train-*.parquet files
        output_dir: Directory to save TSV files
        dataset_name: Name prefix for output files
    """
    print(f"\n{'='*80}")
    print(f"Converting EgoObject Test Set: {dataset_name}")
    print(f"{'='*80}")
    
    parquet_dir = Path(parquet_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all parquet files
    parquet_files = sorted(parquet_dir.glob("train-*.parquet"))
    
    if not parquet_files:
        print(f"Error: No parquet files found in {parquet_dir}")
        return
    
    print(f"Found {len(parquet_files)} parquet files")
    
    # Output file paths
    images_tsv = output_dir / f"{dataset_name}.images.tsv"
    annotations_tsv = output_dir / f"{dataset_name}.annotations.tsv"
    annotations_lineidx = output_dir / f"{dataset_name}.annotations.tsv.lineidx"
    
    sample_count = 0
    error_count = 0
    
    with open(images_tsv, 'wb') as img_tsv, \
         open(annotations_tsv, 'wb') as ann_tsv, \
         open(annotations_lineidx, 'w') as ann_lineidx:
        
        for parquet_file in tqdm(parquet_files, desc=f"Processing {dataset_name}"):
            try:
                df = pd.read_parquet(parquet_file)
                print(f"\n  Processing {parquet_file.name}: {len(df)} samples")
                
                for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"  {parquet_file.name}", leave=False):
                    try:
                        # Load and convert image to base64
                        image = load_image_from_data(row['image'])
                        
                        # Convert to RGB if needed
                        if image.mode != 'RGB':
                            image = image.convert('RGB')
                        
                        # Save to bytes
                        img_buffer = io.BytesIO()
                        image.save(img_buffer, format='JPEG')
                        img_bytes = img_buffer.getvalue()
                        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
                        
                        # Create unique sample ID
                        sample_id = f"egoobject_test_{parquet_file.stem}_{idx}"
                        
                        # Get bbox in absolute coordinates [x, y, w, h] -> [x0, y0, x1, y1]
                        bbox_xywh = row['bbox']
                        x0, y0, w, h = float(bbox_xywh[0]), float(bbox_xywh[1]), float(bbox_xywh[2]), float(bbox_xywh[3])
                        x1, y1 = x0 + w, y0 + h
                        bbox_abs = [x0, y0, x1, y1]
                        
                        # Get intention/phrase
                        # Try different intention fields
                        phrase = None
                        for intention_key in ['intention_1', 'intention_2', 'intention_3', 'scene_reasoning']:
                            if intention_key in row and row[intention_key]:
                                phrase = str(row[intention_key])
                                break
                        
                        if not phrase:
                            phrase = str(row.get('target_category', 'object'))
                        
                        # Create annotation in the format expected by GroundingTSVDataset
                        annotation = {
                            'sample_id': sample_id,
                            'boxes': [
                                {
                                    'phrase': phrase,
                                    'bbox': bbox_abs,  # [x0, y0, x1, y1] in absolute coordinates
                                    'target_category': str(row.get('target_category', 'object'))
                                }
                            ]
                        }
                        
                        # Write image TSV line: {image_id}\t{base64_image}
                        image_line = f"{sample_id}\t{img_base64}\n"
                        image_line_bytes = image_line.encode('utf-8')
                        img_tsv.write(image_line_bytes)
                        
                        # Store the byte offset for the image line (BEFORE writing)
                        # This is the position we need to seek to later
                        img_line_offset = img_tsv.tell() - len(image_line_bytes)
                        
                        # Write annotation TSV line: {image_byte_offset}\t{annotation_json}
                        annotation_line = f"{img_line_offset}\t{json.dumps(annotation, ensure_ascii=False)}\n"
                        annotation_line_bytes = annotation_line.encode('utf-8')
                        
                        # Store annotation line offset BEFORE writing
                        ann_line_offset_for_idx = ann_tsv.tell()
                        
                        ann_tsv.write(annotation_line_bytes)
                        
                        # Write annotation line index (this is the offset of the annotation line itself)
                        ann_lineidx.write(f"{ann_line_offset_for_idx}\n")
                        
                        sample_count += 1
                        
                    except Exception as e:
                        print(f"\n    Error processing sample {idx} in {parquet_file.name}: {e}")
                        error_count += 1
                        continue
                
            except Exception as e:
                print(f"\n  Error loading {parquet_file.name}: {e}")
                error_count += 1
                continue
    
    print(f"\n{'='*80}")
    print(f"Conversion Complete!")
    print(f"{'='*80}")
    print(f"Total samples converted: {sample_count}")
    print(f"Errors encountered: {error_count}")
    print(f"\nOutput files:")
    print(f"  - {images_tsv}")
    print(f"  - {annotations_tsv}")
    print(f"  - {annotations_lineidx}")
    print(f"{'='*80}\n")
    
    return {
        'dataset': dataset_name,
        'total_samples': sample_count,
        'errors': error_count,
        'images_tsv': str(images_tsv),
        'annotations_tsv': str(annotations_tsv),
        'annotations_lineidx': str(annotations_lineidx)
    }


def visualize_samples(output_dir, dataset_name, num_samples=5):
    """Visualize some samples to verify conversion"""
    from PIL import ImageDraw
    
    print(f"\nVisualizing {num_samples} random samples...")
    
    output_dir = Path(output_dir)
    viz_dir = output_dir / "visualizations" / dataset_name
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    images_tsv = output_dir / f"{dataset_name}.images.tsv"
    annotations_tsv = output_dir / f"{dataset_name}.annotations.tsv"
    annotations_lineidx = output_dir / f"{dataset_name}.annotations.tsv.lineidx"
    
    # Read line offsets
    with open(annotations_lineidx, 'r') as f:
        anno_line_offsets = [int(line.strip()) for line in f]
    
    # Sample random indices
    import random
    num_samples = min(num_samples, len(anno_line_offsets))
    sample_indices = random.sample(range(len(anno_line_offsets)), num_samples)
    
    for idx in sample_indices:
        try:
            # Read annotation
            with open(annotations_tsv, 'rb') as f:
                f.seek(anno_line_offsets[idx])
                line = f.readline().decode('utf-8').strip()
                img_byte_offset_str, ann_json = line.split('\t')
                img_byte_offset = int(img_byte_offset_str)
                annotation = json.loads(ann_json)
            
            # Read image
            with open(images_tsv, 'rb') as f:
                f.seek(img_byte_offset)
                img_line = f.readline().decode('utf-8').strip()
                sample_id, img_base64 = img_line.split('\t')
                img_bytes = base64.b64decode(img_base64)
                image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            
            # Draw bounding boxes
            draw = ImageDraw.Draw(image)
            for box_data in annotation['boxes']:
                bbox = box_data['bbox']
                phrase = box_data['phrase']
                
                # Draw box
                draw.rectangle(bbox, outline='red', width=3)
                
                # Draw label
                draw.text((bbox[0], bbox[1] - 15), phrase[:50], fill='red')
            
            # Save visualization
            viz_path = viz_dir / f"{annotation['sample_id']}.jpg"
            image.save(viz_path)
            print(f"  Saved: {viz_path}")
            
        except Exception as e:
            print(f"  Error visualizing sample {idx}: {e}")
    
    print(f"\nVisualization complete! Check: {viz_dir}")


def main():
    # Configuration
    parquet_dir = "/home/hairong/hairong/data/egoobject-intention-eval/data"
    output_dir = "/home/hairong/hairong/data/intention_datasets_tsv_fixed"
    dataset_name = "egoobject_test"
    
    print("="*80)
    print("EgoObject Test Set to TSV Converter")
    print("="*80)
    print(f"Input:  {parquet_dir}")
    print(f"Output: {output_dir}")
    print("="*80)
    
    # Convert
    result = convert_egoobject_test_to_tsv(
        parquet_dir=parquet_dir,
        output_dir=output_dir,
        dataset_name=dataset_name
    )
    
    # Visualize some samples
    visualize_samples(output_dir, dataset_name, num_samples=10)
    
    # Print summary
    print("\n" + "="*80)
    print("CONVERSION SUMMARY")
    print("="*80)
    print(json.dumps(result, indent=2))
    print("="*80)


if __name__ == "__main__":
    main()

