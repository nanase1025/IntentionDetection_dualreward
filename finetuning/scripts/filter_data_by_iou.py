#!/usr/bin/env python3
"""
Simplified version: Filter training data based on IoU scores from SFT model predictions.
This script:
1. Loads the SFT checkpoint
2. Runs inference on all training samples
3. Calculates IoU for each sample
4. Saves IoU scores to JSON
5. Filters samples based on IoU thresholds and creates new TSV files
"""

import os
import sys
import json
import argparse
import base64
from pathlib import Path
from typing import List, Dict, Tuple
import torch
from tqdm import tqdm
from PIL import Image
import re
import io

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """Calculate IoU between two boxes in xyxy format"""
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def parse_detection_output(text: str, width: int, height: int) -> List[List[float]]:
    """Parse model output to extract bounding boxes"""
    try:
        text = text.replace("\n", "").strip()
        boxes = []
        
        # Pattern to match <box>[[x0,y0,x1,y1]]</box>
        box_pattern = r'<box>\[\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\]</box>'
        matches = re.findall(box_pattern, text)
        
        for match in matches:
            x0, y0, x1, y1 = map(int, match)
            # Convert from bins to coordinates
            x0 = (x0 / 999) * width
            y0 = (y0 / 999) * height
            x1 = (x1 / 999) * width
            y1 = (y1 / 999) * height
            boxes.append([x0, y0, x1, y1])
        
        return boxes
    except Exception as e:
        return []


def load_tsv_sample(img_tsv_file: str, ann_tsv_file: str, ann_lineidx_file: str, idx: int) -> Dict:
    """Load a single sample from TSV files"""
    # Read annotation line index
    with open(ann_lineidx_file, 'r') as f:
        line_offsets = [int(line.strip()) for line in f]
    
    # Read annotation
    with open(ann_tsv_file, 'r') as f:
        f.seek(line_offsets[idx])
        img_line_idx, ann_json = f.readline().strip().split('\t')
        img_line_idx = int(img_line_idx)
        annotation = json.loads(ann_json)
    
    # Read image
    with open(img_tsv_file, 'r') as f:
        lines = f.readlines()
        _, img_base64 = lines[img_line_idx].strip().split('\t')
        img_bytes = base64.b64decode(img_base64)
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    
    return {
        'image': image,
        'annotation': annotation,
        'img_line_idx': img_line_idx,
    }


def evaluate_sample(model, processor, image: Image.Image, annotation: Dict, device: str) -> Tuple[float, str]:
    """Evaluate a single sample and return IoU score and prediction"""
    
    # Get phrase from annotation
    if 'boxes' in annotation and len(annotation['boxes']) > 0:
        phrase = annotation['boxes'][0].get('phrase', 'object')
    else:
        return 0.0, ""
    
    # Prepare prompt
    prompt = f"Please provide the bounding box coordinate of the region this sentence describes: {phrase}"
    
    # Prepare messages for model
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    
    # Prepare for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)
    
    # Generate
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=256, temperature=0.0)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )[0]
    
    # Parse ground truth boxes
    gt_boxes = []
    for box_data in annotation['boxes']:
        gt_boxes.append(box_data['bbox'])
    
    # Get image dimensions
    img_width, img_height = image.size
    
    # Parse prediction boxes
    pred_boxes = parse_detection_output(output_text, img_width, img_height)
    
    # Calculate IoU (take max IoU if multiple predictions)
    max_iou = 0.0
    if len(pred_boxes) > 0 and len(gt_boxes) > 0:
        for pred_box in pred_boxes:
            for gt_box in gt_boxes:
                iou = calculate_iou(pred_box, gt_box)
                max_iou = max(max_iou, iou)
    
    return max_iou, output_text


def evaluate_dataset(
    model,
    processor,
    dataset_config: Dict,
    device: str,
    output_dir: Path,
    dataset_name: str
) -> List[Tuple[int, float]]:
    """Evaluate entire dataset and return list of (sample_idx, iou_score)"""
    
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
            
            iou, pred = evaluate_sample(
                model, processor, sample['image'], sample['annotation'], device
            )
            results.append((idx, iou))
            iou_scores.append(iou)
            
            # Log some examples
            if idx < 5 or (idx % 100 == 0):
                print(f"\nSample {idx}: IoU = {iou:.4f}")
                if idx < 3:
                    print(f"Prediction: {pred[:150]}...")
        except Exception as e:
            print(f"Error evaluating sample {idx}: {e}")
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


def filter_and_create_tsv(
    original_config: Dict,
    iou_results: List[Tuple[int, float]],
    output_dir: Path,
    dataset_name: str,
    min_iou: float,
    max_iou: float
) -> None:
    """Filter samples based on IoU threshold and create new TSV files"""
    
    print(f"\n{'='*80}")
    print(f"Filtering {dataset_name} with IoU range [{min_iou}, {max_iou}]")
    print(f"{'='*80}")
    
    # Get indices that pass the filter
    filtered_indices = [idx for idx, iou in iou_results if min_iou <= iou <= max_iou]
    
    print(f"Original samples: {len(iou_results)}")
    print(f"Filtered samples: {len(filtered_indices)} ({len(filtered_indices)/len(iou_results)*100:.1f}%)")
    
    if len(filtered_indices) == 0:
        print(f"Warning: No samples pass the filter for {dataset_name}!")
        return
    
    # Read annotation line offsets
    with open(original_config["anno_idx"], 'r') as f:
        anno_line_offsets = [int(line.strip()) for line in f]
    
    # Create new TSV files
    output_image_tsv = output_dir / f"{dataset_name}_filtered.images.tsv"
    output_anno_tsv = output_dir / f"{dataset_name}_filtered.annotations.tsv"
    output_anno_idx = output_dir / f"{dataset_name}_filtered.annotations.tsv.lineidx"
    
    # Open files in binary mode for byte offset reading
    img_tsv_file = open(original_config["image_tsv"], 'rb')
    ann_tsv_file = open(original_config["anno_tsv"], 'rb')
    
    with open(output_image_tsv, 'wb') as img_out_f, \
         open(output_anno_tsv, 'wb') as ann_out_f, \
         open(output_anno_idx, 'w') as idx_out_f:
        
        new_ann_offset = 0
        new_img_offset = 0
        
        for old_idx in filtered_indices:
            # Read annotation line using byte offset
            ann_tsv_file.seek(anno_line_offsets[old_idx])
            ann_line_bytes = ann_tsv_file.readline()
            ann_line_str = ann_line_bytes.decode('utf-8').strip()
            
            # Parse to get image byte offset
            img_byte_offset_str, ann_json = ann_line_str.split('\t', 1)
            img_byte_offset = int(img_byte_offset_str)
            
            # Read image line using byte offset
            img_tsv_file.seek(img_byte_offset)
            img_line_bytes = img_tsv_file.readline()
            
            # Write image to new file
            img_out_f.write(img_line_bytes)
            
            # Write annotation with NEW image byte offset
            new_ann_line = f"{new_img_offset}\t{ann_json}\n"
            new_ann_line_bytes = new_ann_line.encode('utf-8')
            ann_out_f.write(new_ann_line_bytes)
            
            # Write annotation line index (byte offset)
            idx_out_f.write(f"{new_ann_offset}\n")
            
            # Update offsets
            new_img_offset += len(img_line_bytes)
            new_ann_offset += len(new_ann_line_bytes)
    
    img_tsv_file.close()
    ann_tsv_file.close()
    
    print(f"Created filtered TSV files:")
    print(f"  Images: {output_image_tsv}")
    print(f"  Annotations: {output_anno_tsv}")
    print(f"  Index: {output_anno_idx}")


def main():
    parser = argparse.ArgumentParser(description="Filter training data by IoU scores")
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to SFT checkpoint (required for evaluation)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="work_dirs/filtered_data",
        help="Output directory for filtered data"
    )
    parser.add_argument(
        "--min_iou",
        type=float,
        default=0.2,
        help="Minimum IoU threshold (default: 0.2)"
    )
    parser.add_argument(
        "--max_iou",
        type=float,
        default=0.7,
        help="Maximum IoU threshold (default: 0.7)"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=2,
        help="GPU device ID (default: 2)"
    )
    parser.add_argument(
        "--evaluate_only",
        action="store_true",
        help="Only evaluate and save IoU scores, don't filter"
    )
    parser.add_argument(
        "--filter_only",
        action="store_true",
        help="Only filter using existing IoU scores, don't evaluate"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.evaluate_only and args.filter_only:
        print("Error: Cannot use both --evaluate_only and --filter_only")
        return
    
    if not args.filter_only and not args.model_path:
        print("Error: --model_path is required for evaluation")
        return
    
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
    
    device = f"cuda:{args.gpu}" if "CUDA_VISIBLE_DEVICES" not in os.environ else "cuda:0"
    
    # Step 1: Evaluation (if needed)
    all_results = {}
    
    if not args.filter_only:
        print("="*80)
        print("STEP 1: EVALUATION")
        print("="*80)
        print("Loading model...")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )
        model.eval()
        
        processor = AutoProcessor.from_pretrained(
            args.model_path,
            trust_remote_code=True,
        )
        
        print(f"Model loaded from: {args.model_path}")
        print(f"Using device: {device}")
        
        # Evaluate all datasets
        for dataset_name, dataset_config in datasets.items():
            results = evaluate_dataset(
                model, processor, dataset_config, device, output_dir, dataset_name
            )
            all_results[dataset_name] = results
        
        if args.evaluate_only:
            print(f"\n{'='*80}")
            print("Evaluation complete! IoU scores saved.")
            print(f"{'='*80}")
            print(f"\nTo filter data, run:")
            print(f"  bash scripts/apply_filter.sh {args.min_iou} {args.max_iou}")
            return
    
    # Step 2: Filtering (if needed)
    if not args.evaluate_only:
        print(f"\n{'='*80}")
        print("STEP 2: FILTERING")
        print(f"{'='*80}")
        
        # Load existing results if we skipped evaluation
        if args.filter_only:
            print("Loading existing IoU results...")
            for dataset_name in datasets.keys():
                results_file = output_dir / f"{dataset_name}_iou_scores.json"
                if not results_file.exists():
                    print(f"Error: Results file not found: {results_file}")
                    print("Please run evaluation first: bash scripts/run_filter_data.sh")
                    return
                with open(results_file, 'r') as f:
                    data = json.load(f)
                    all_results[dataset_name] = [(r["idx"], r["iou"]) for r in data["results"]]
        
        # Filter datasets
        print(f"\nFiltering datasets with IoU range [{args.min_iou}, {args.max_iou}]")
        
        for dataset_name, results in all_results.items():
            filter_and_create_tsv(
                datasets[dataset_name],
                results,
                output_dir,
                dataset_name,
                args.min_iou,
                args.max_iou
            )
        
        print(f"\n{'='*80}")
        print("Filtering complete!")
        print(f"{'='*80}")
        print(f"\nFiltered data saved to: {output_dir}")
        print(f"\nNext steps:")
        print(f"1. Update GRPO config to use filtered TSV files")
        print(f"2. Run GRPO training with filtered data")


if __name__ == "__main__":
    main()
