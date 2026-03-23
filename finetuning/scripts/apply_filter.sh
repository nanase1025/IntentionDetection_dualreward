#!/bin/bash

# Step 2: Apply filtering with specific IoU thresholds
# This script is FAST and can be run multiple times with different thresholds
# Prerequisites: Must run run_filter_data.sh first to generate IoU scores

set -e

# Parse arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: bash scripts/apply_filter.sh MIN_IOU MAX_IOU"
    echo "Example: bash scripts/apply_filter.sh 0.2 0.7"
    echo ""
    echo "This will filter samples where MIN_IOU <= IoU <= MAX_IOU"
    exit 1
fi

MIN_IOU=$1
MAX_IOU=$2
OUTPUT_DIR="work_dirs/filtered_data"

echo "========================================="
echo "Applying IoU Filter to Training Data"
echo "========================================="
echo "IoU Range: [${MIN_IOU}, ${MAX_IOU}]"
echo "Output: ${OUTPUT_DIR}"
echo ""
echo "This is FAST - no model evaluation needed!"
echo "========================================="
echo ""

# Check if IoU scores exist
if [ ! -f "${OUTPUT_DIR}/coco_outdoor_train_iou_scores.json" ]; then
    echo "ERROR: IoU scores not found!"
    echo "Please run evaluation first:"
    echo "  bash scripts/run_filter_data.sh"
    exit 1
fi

# Activate virtual environment
source ~/.pyenv/versions/rexomni/bin/activate

cd /home/hairong/hairong/code/IntentionDetection/finetuning

# Run filtering only (no evaluation)
python scripts/filter_data_by_iou.py \
    --output_dir ${OUTPUT_DIR} \
    --min_iou ${MIN_IOU} \
    --max_iou ${MAX_IOU} \
    --filter_only

echo ""
echo "========================================="
echo "Filtering Complete!"
echo "========================================="
echo "Filtered TSV files saved to: ${OUTPUT_DIR}"
echo ""
echo "Filter statistics:"
python3 << EOF
import json
from pathlib import Path

output_dir = Path("${OUTPUT_DIR}")
total_original = 0
total_filtered = 0

for dataset in ["coco_outdoor_train", "scannet_train", "egoobject_train"]:
    results_file = output_dir / f"{dataset}_iou_scores.json"
    with open(results_file) as f:
        data = json.load(f)
    
    total = data["total_samples"]
    filtered = sum(1 for r in data["results"] if ${MIN_IOU} <= r["iou"] <= ${MAX_IOU})
    
    total_original += total
    total_filtered += filtered
    
    print(f"{dataset}:")
    print(f"  Original: {total} samples")
    print(f"  Filtered: {filtered} samples ({filtered/total*100:.1f}%)")
    print()

print(f"TOTAL:")
print(f"  Original: {total_original} samples")
print(f"  Filtered: {total_filtered} samples ({total_filtered/total_original*100:.1f}%)")
EOF
echo ""
echo "Next steps:"
echo "1. If you want different thresholds, run this again:"
echo "   bash scripts/apply_filter.sh 0.3 0.6"
echo ""
echo "2. When satisfied, update GRPO config and train:"
echo "   configs/grpo_intention_datasets_filtered.py (already configured)"
echo "   bash scripts/grpo_intention_datasets.sh"
echo "========================================="

