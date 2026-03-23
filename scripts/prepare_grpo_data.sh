#!/bin/bash

# Complete workflow for GRPO data preparation
# This script:
# 1. Evaluates SFT model on training sets
# 2. Analyzes IoU distribution
# 3. Filters training data based on IoU threshold

set -e

# ========================================
# Configuration
# ========================================
MODEL_PATH="/home/hairong/hairong/code/IntentionDetection/finetuning/work_dirs/intention_sft"
EVAL_OUTPUT="evaluation_three_datasets_train_results.json"
FILTER_OUTPUT_DIR="finetuning/work_dirs/filtered_data_grpo"
GPU=2
BACKEND="vllm"

# IoU threshold for filtering (adjust based on distribution analysis)
IOU_THRESHOLD=0.3

# For quick testing, uncomment to limit samples
# MAX_SAMPLES="100"

echo "========================================="
echo "GRPO DATA PREPARATION WORKFLOW"
echo "========================================="
echo "Model: ${MODEL_PATH}"
echo "IoU Threshold: ${IOU_THRESHOLD}"
echo "Output: ${FILTER_OUTPUT_DIR}"
echo "========================================="

# Activate environment
source ~/.pyenv/versions/rexomni/bin/activate
cd /home/hairong/hairong/code/IntentionDetection

# ========================================
# Step 1: Evaluate on training sets
# ========================================
echo ""
echo "========================================="
echo "STEP 1: Evaluate SFT model on TRAIN sets"
echo "========================================="

export CUDA_VISIBLE_DEVICES=${GPU}

if [ -n "$MAX_SAMPLES" ]; then
    python3 evaluate_intention_three_datasets_train.py \
        --checkpoint ${MODEL_PATH} \
        --output_file ${EVAL_OUTPUT} \
        --backend ${BACKEND} \
        --max_samples ${MAX_SAMPLES}
else
    python3 evaluate_intention_three_datasets_train.py \
        --checkpoint ${MODEL_PATH} \
        --output_file ${EVAL_OUTPUT} \
        --backend ${BACKEND}
fi

echo ""
echo "✅ Evaluation complete: ${EVAL_OUTPUT}"

# ========================================
# Step 2: Analyze IoU distribution
# ========================================
echo ""
echo "========================================="
echo "STEP 2: Analyze IoU Distribution"
echo "========================================="

python3 filter_train_data_by_iou.py \
    --eval_results ${EVAL_OUTPUT} \
    --iou_threshold ${IOU_THRESHOLD} \
    --analyze_only

echo ""
echo "📊 Review the IoU distribution above."
echo "If you want to adjust the threshold, edit IOU_THRESHOLD in this script."
echo ""
read -p "Continue with filtering at IoU >= ${IOU_THRESHOLD}? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted. Adjust IOU_THRESHOLD and re-run."
    exit 0
fi

# ========================================
# Step 3: Filter training data
# ========================================
echo ""
echo "========================================="
echo "STEP 3: Filter Training Data"
echo "========================================="

python3 filter_train_data_by_iou.py \
    --eval_results ${EVAL_OUTPUT} \
    --iou_threshold ${IOU_THRESHOLD} \
    --output_dir ${FILTER_OUTPUT_DIR}

echo ""
echo "✅ Filtered datasets saved to: ${FILTER_OUTPUT_DIR}"

# ========================================
# Step 4: Summary
# ========================================
echo ""
echo "========================================="
echo "WORKFLOW COMPLETE!"
echo "========================================="
echo ""
echo "📁 Output files:"
echo "   1. Evaluation results: ${EVAL_OUTPUT}"
echo "   2. Filtered datasets:  ${FILTER_OUTPUT_DIR}/"
echo "   3. Filtering summary:  ${FILTER_OUTPUT_DIR}/filtering_summary.json"
echo ""
echo "📝 Next steps:"
echo "   1. Update GRPO config to use filtered datasets"
echo "   2. Run GRPO training"
echo ""
echo "Example GRPO config paths:"
echo "   coco_outdoor: ${FILTER_OUTPUT_DIR}/coco_outdoor/train_filtered.*.tsv"
echo "   scannet:      ${FILTER_OUTPUT_DIR}/scannet/train_filtered.*.tsv"
echo "   egoobject:    ${FILTER_OUTPUT_DIR}/egoobject/train_filtered.*.tsv"
echo "========================================="


