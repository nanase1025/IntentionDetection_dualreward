#!/usr/bin/env python3
"""
Evaluate SFT checkpoint on intention datasets and save IoU scores for filtering

Usage:
    python evaluate_and_save_iou.py --checkpoint work_dirs/intention_sft/checkpoint-2523
"""

import argparse
import json
import os
import re
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import base64
import io

import sys
sys.path.insert(0, str(Path(__file__).parent))
from rex_omni import RexOmniWrapper


def compute_iou(box1, box2):
    """Compute IoU of two bboxes (in absolute coordinates)"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Compute intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # Compute union
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def parse_prediction_bboxes(response_text):
    """Parse bbox from model response"""
    # Pattern: <box>[[x0,y0,x1,y1]]</box>
    pattern = r'<box>\[\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\]</box>'
    matches = re.findall(pattern, response_text)
    
    bboxes = []
    for match in matches:
        x0, y0, x1, y1 = map(int, match)
        bboxes.append([x0, y0, x1, y1])
    
    return bboxes


def denormalize_box(norm_box, img_width, img_height):
    """Convert normalized coordinates [0,999] to absolute coordinates"""
    x0 = (norm_box[0] / 999.0) * img_width
    y0 = (norm_box[1] / 999.0) * img_height
    x1 = (norm_box[2] / 999.0) * img_width
    y1 = (norm_box[3] / 999.0) * img_height
    return [x0, y0, x1, y1]


def load_tsv_sample(img_tsv_file, ann_tsv_file, ann_lineidx_file, idx, img_line_index=None):
    """Load a single sample from TSV files
    
    Note: The first column in annotations.tsv is the BYTE OFFSET (not line index) 
    of the corresponding image in images.tsv
    """
    # Read annotation line index
    with open(ann_lineidx_file, 'r') as f:
        line_offsets = [int(line.strip()) for line in f]
    
    if idx >= len(line_offsets):
        raise IndexError(f"Sample index {idx} out of range (total: {len(line_offsets)})")
    
    # Read annotation
    with open(ann_tsv_file, 'r') as f:
        f.seek(line_offsets[idx])
        line = f.readline().strip()
        if not line:
            raise ValueError(f"Empty annotation line at index {idx}")
        parts = line.split('\t')
        if len(parts) != 2:
            raise ValueError(f"Invalid annotation format at index {idx}: expected 2 parts, got {len(parts)}")
        img_byte_offset, ann_json = parts
        img_byte_offset = int(img_byte_offset)  # This is BYTE OFFSET, not line number
        annotation = json.loads(ann_json)
    
    # Read image using BYTE OFFSET (not line index)
    with open(img_tsv_file, 'r') as f:
        f.seek(img_byte_offset)  # Seek to byte offset directly
        line = f.readline().strip()
        parts = line.split('\t')
        if len(parts) != 2:
            raise ValueError(f"Invalid image format at byte offset {img_byte_offset}")
        _, img_base64 = parts
    
    img_bytes = base64.b64decode(img_base64)
    image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    
    return {
        'image': image,
        'annotation': annotation,
        'img_byte_offset': img_byte_offset,
    }


def evaluate_dataset(model, dataset_config, dataset_name, output_dir):
    """Evaluate a single dataset and save IoU scores"""
    print(f"\n{'='*80}")
    print(f"Evaluating {dataset_name}")
    print(f"{'='*80}")
    
    # Count total samples
    with open(dataset_config["anno_idx"], 'r') as f:
        total_samples = len(f.readlines())
    
    print(f"Total samples: {total_samples}")
    
    results = []
    iou_scores = []
    
    # Evaluate each sample
    for idx in tqdm(range(total_samples), desc=f"Evaluating {dataset_name}"):
        try:
            sample = load_tsv_sample(
                dataset_config["image_tsv"],
                dataset_config["anno_tsv"],
                dataset_config["anno_idx"],
                idx
            )
            
            # Get phrase from annotation
            if 'boxes' in sample['annotation'] and len(sample['annotation']['boxes']) > 0:
                phrase = sample['annotation']['boxes'][0].get('phrase', 'object')
                gt_boxes_absolute = [box_data['bbox'] for box_data in sample['annotation']['boxes']]
            else:
                results.append((idx, 0.0))
                iou_scores.append(0.0)
                continue
            
            # Run inference using detection task
            inference_results = model.inference(
                images=sample['image'],
                task="detection",
                categories=[phrase]  # Use the phrase as the category to detect
            )
            
            # Extract predicted boxes
            pred_boxes_abs = []
            if inference_results and len(inference_results) > 0:
                result = inference_results[0]
                if result.get('success', False):
                    predictions = result.get('extracted_predictions', {})
                    # Get predictions for the phrase
                    for pred in predictions.get(phrase, []):
                        coords = pred.get('coords', [])
                        if isinstance(coords, (list, tuple)) and len(coords) == 4:
                            pred_boxes_abs.append(coords)
            
            # Compute max IoU
            max_iou = 0.0
            if len(pred_boxes_abs) > 0 and len(gt_boxes_absolute) > 0:
                for pred_box in pred_boxes_abs:
                    for gt_box in gt_boxes_absolute:
                        iou = compute_iou(pred_box, gt_box)
                        max_iou = max(max_iou, iou)
            
            results.append((idx, max_iou))
            iou_scores.append(max_iou)
            
            # Log some examples
            if idx < 5 or (idx % 100 == 0):
                print(f"\nSample {idx}: IoU = {max_iou:.4f}")
                if idx < 3:
                    print(f"  Phrase: {phrase}")
                    print(f"  Predicted {len(pred_boxes_abs)} boxes")
                    if inference_results and len(inference_results) > 0:
                        print(f"  Raw output: {inference_results[0].get('raw_output', '')[:150]}...")
                    
        except Exception as e:
            print(f"\nError evaluating sample {idx}: {e}")
            results.append((idx, 0.0))
            iou_scores.append(0.0)
    
    # Save results
    results_file = output_dir / f"{dataset_name}_iou_scores.json"
    with open(results_file, 'w') as f:
        json.dump({
            "dataset": dataset_name,
            "total_samples": len(results),
            "results": [{"idx": idx, "iou": iou} for idx, iou in results],
            "statistics": {
                "mean_iou": sum(iou_scores) / len(iou_scores) if iou_scores else 0.0,
                "max_iou": max(iou_scores) if iou_scores else 0.0,
                "min_iou": min(iou_scores) if iou_scores else 0.0,
            }
        }, f, indent=2)
    
    print(f"\n{dataset_name} Statistics:")
    print(f"  Mean IoU: {sum(iou_scores) / len(iou_scores):.4f}")
    print(f"  Max IoU: {max(iou_scores):.4f}")
    print(f"  Min IoU: {min(iou_scores):.4f}")
    print(f"  Results saved to: {results_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate SFT model and save IoU scores")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to SFT checkpoint")
    parser.add_argument("--output_dir", type=str, default="work_dirs/filtered_data", help="Output directory")
    parser.add_argument("--backend", type=str, default="transformers", choices=["transformers", "vllm"], help="Inference backend")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Dataset configurations
    datasets = {
        "coco_outdoor_train": {
            "image_tsv": "/home/hairong/hairong/data/intention_datasets_tsv_fixed/coco_outdoor_train.images.tsv",
            "anno_tsv": "/home/hairong/hairong/data/intention_datasets_tsv_fixed/coco_outdoor_train.annotations.tsv",
            "anno_idx": "/home/hairong/hairong/data/intention_datasets_tsv_fixed/coco_outdoor_train.annotations.tsv.lineidx",
        },
        "scannet_train": {
            "image_tsv": "/home/hairong/hairong/data/intention_datasets_tsv_fixed/scannet_train.images.tsv",
            "anno_tsv": "/home/hairong/hairong/data/intention_datasets_tsv_fixed/scannet_train.annotations.tsv",
            "anno_idx": "/home/hairong/hairong/data/intention_datasets_tsv_fixed/scannet_train.annotations.tsv.lineidx",
        },
        "egoobject_train": {
            "image_tsv": "/home/hairong/hairong/data/intention_datasets_tsv_fixed/egoobject_train.images.tsv",
            "anno_tsv": "/home/hairong/hairong/data/intention_datasets_tsv_fixed/egoobject_train.annotations.tsv",
            "anno_idx": "/home/hairong/hairong/data/intention_datasets_tsv_fixed/egoobject_train.annotations.tsv.lineidx",
        },
    }
    
    # Load model
    print(f"Loading model from: {args.checkpoint}")
    print(f"Backend: {args.backend}")
    
    model = RexOmniWrapper(
        model_path=args.checkpoint,
        backend=args.backend,
    )
    
    print("Model loaded successfully!")
    
    # Evaluate all datasets
    for dataset_name, dataset_config in datasets.items():
        evaluate_dataset(model, dataset_config, dataset_name, output_dir)
    
    print(f"\n{'='*80}")
    print("All datasets evaluated!")
    print(f"{'='*80}")
    print(f"\nResults saved to: {output_dir}")
    print(f"\nNext steps:")
    print(f"1. Analyze IoU distribution:")
    print(f"   python finetuning/scripts/analyze_iou_distribution.py")
    print(f"2. Apply filtering with desired thresholds:")
    print(f"   bash finetuning/scripts/apply_filter.sh 0.2 0.7")


if __name__ == "__main__":
    main()

