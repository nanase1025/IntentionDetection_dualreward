#!/usr/bin/env python3
"""
Filter training data based on IoU scores for GRPO training.
Keep only samples with IoU < 0.8 (poorly predicted samples that need improvement).
"""

import json
import os
import argparse
from tqdm import tqdm


def load_evaluation_results(json_file, iou_threshold):
    """Load evaluation results and extract sample IDs with IoU < threshold."""
    print(f"Loading evaluation results from: {json_file}")
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    filtered_samples = {}
    
    for dataset_name in ['coco_outdoor', 'scannet', 'egoobject']:
        if dataset_name in data['detailed_predictions']:
            predictions = data['detailed_predictions'][dataset_name]
            filtered_samples[dataset_name] = []
            
            for sample_id, sample_data in predictions.items():
                iou = sample_data.get('iou', 0.0)
                if iou < iou_threshold:  # Keep samples that need improvement
                    # Extract numeric ID from "sample_N"
                    numeric_id = int(sample_id.split('_')[1])
                    filtered_samples[dataset_name].append(numeric_id)
            
            filtered_samples[dataset_name].sort()
            
            print(f"  {dataset_name}: {len(filtered_samples[dataset_name])} samples (IoU < {iou_threshold})")
    
    return filtered_samples


def filter_tsv_files(dataset_name, filtered_sample_ids, input_base_dir, output_base_dir):
    """Filter TSV files to keep only specified sample IDs (line indices in annotation file)."""
    
    # Construct file paths
    img_tsv_in = os.path.join(input_base_dir, f"{dataset_name}_train.images.tsv")
    ann_tsv_in = os.path.join(input_base_dir, f"{dataset_name}_train.annotations.tsv")
    ann_lineidx_in = os.path.join(input_base_dir, f"{dataset_name}_train.annotations.tsv.lineidx")
    
    img_tsv_out = os.path.join(output_base_dir, f"{dataset_name}_train_grpo.images.tsv")
    ann_tsv_out = os.path.join(output_base_dir, f"{dataset_name}_train_grpo.annotations.tsv")
    ann_lineidx_out = os.path.join(output_base_dir, f"{dataset_name}_train_grpo.annotations.tsv.lineidx")
    
    print(f"\nProcessing {dataset_name}...")
    print(f"  Input images: {img_tsv_in}")
    print(f"  Input annotations: {ann_tsv_in}")
    print(f"  Output images: {img_tsv_out}")
    print(f"  Output annotations: {ann_tsv_out}")
    
    # Convert list to set for faster lookup
    filtered_ids_set = set(filtered_sample_ids)
    
    # First pass: collect all image byte offsets that we need
    print("  Collecting required image byte offsets...")
    required_image_offsets = set()
    filtered_annotations = []
    
    with open(ann_tsv_in, 'r') as f:
        for idx, line in enumerate(tqdm(f, desc="  Reading annotations")):
            if idx in filtered_ids_set:
                parts = line.rstrip('\n').split('\t', 1)
                if len(parts) >= 2:
                    image_byte_offset = int(parts[0])
                    required_image_offsets.add(image_byte_offset)
                    filtered_annotations.append((image_byte_offset, line))
    
    print(f"  Filtered: {len(filtered_annotations)} annotations")
    print(f"  Unique images needed: {len(required_image_offsets)}")
    
    # Second pass: read only the required images
    print("  Loading required images...")
    images_dict = {}  # Map old byte offset -> image line
    
    with open(img_tsv_in, 'rb') as f:
        current_offset = 0
        for line in tqdm(f, desc="  Reading images"):
            if current_offset in required_image_offsets:
                images_dict[current_offset] = line
            current_offset += len(line)
    
    print(f"  Loaded {len(images_dict)} unique images")
    
    # Write filtered images and build offset mapping
    print("  Writing filtered images...")
    new_image_byte_offset = 0
    old_to_new_offset = {}  # Map old byte offset -> new byte offset
    
    with open(img_tsv_out, 'wb') as f:
        for old_offset in tqdm(sorted(required_image_offsets), desc="  Writing images"):
            if old_offset in images_dict:
                old_to_new_offset[old_offset] = new_image_byte_offset
                image_line = images_dict[old_offset]
                f.write(image_line)
                new_image_byte_offset += len(image_line)
    
    # Write filtered annotations with updated byte offsets
    print("  Writing filtered annotations...")
    ann_byte_offset = 0
    ann_offsets = [0]  # Line index for annotations
    
    with open(ann_tsv_out, 'wb') as f:
        for old_img_offset, original_line in tqdm(filtered_annotations, desc="  Writing annotations"):
            if old_img_offset in old_to_new_offset:
                new_img_offset = old_to_new_offset[old_img_offset]
                parts = original_line.rstrip('\n').split('\t', 1)
                if len(parts) >= 2:
                    # Update byte offset to point to new image location
                    new_line = f"{new_img_offset}\t{parts[1]}\n"
                    line_bytes = new_line.encode('utf-8')
                    f.write(line_bytes)
                    ann_byte_offset += len(line_bytes)
                    ann_offsets.append(ann_byte_offset)
    
    # Write line index file for annotations
    print("  Writing line index...")
    with open(ann_lineidx_out, 'w') as f:
        for offset in ann_offsets[:-1]:  # Don't include the last offset
            f.write(f"{offset}\n")
    
    print(f"  ✅ Filtered dataset saved:")
    print(f"     Images: {len(images_dict)} samples")
    print(f"     Annotations: {len(filtered_annotations)} samples")


def main():
    parser = argparse.ArgumentParser(description='Filter training data for GRPO based on IoU scores')
    parser.add_argument('--eval_json', type=str, required=True,
                        help='Evaluation results JSON file')
    parser.add_argument('--input_dir', type=str,
                        default='/home/hairong/hairong/data/intention_datasets_tsv_fixed',
                        help='Input TSV directory')
    parser.add_argument('--output_dir', type=str,
                        default='/home/hairong/hairong/data/intention_datasets_tsv_fixed_grpo',
                        help='Output TSV directory for filtered data')
    parser.add_argument('--iou_threshold', type=float, default=0.8,
                        help='Keep samples with IoU < threshold (default: 0.8)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 70)
    print("GRPO Data Filtering")
    print("=" * 70)
    print(f"IoU threshold: < {args.iou_threshold}")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 70)
    
    # Load evaluation results
    filtered_samples = load_evaluation_results(args.eval_json, args.iou_threshold)
    
    total_filtered = sum(len(ids) for ids in filtered_samples.values())
    print(f"\nTotal samples to filter: {total_filtered}")
    
    # Filter each dataset
    for dataset_name, sample_ids in filtered_samples.items():
        if len(sample_ids) > 0:
            filter_tsv_files(dataset_name, sample_ids, args.input_dir, args.output_dir)
    
    print("\n" + "=" * 70)
    print("✅ Filtering complete!")
    print("=" * 70)
    print(f"\nFiltered data saved to: {args.output_dir}")
    print("\nNext steps:")
    print("  1. Update training config to use the filtered data")
    print("  2. Run GRPO training on these poorly predicted samples")
    print("=" * 70)


if __name__ == '__main__':
    main()
