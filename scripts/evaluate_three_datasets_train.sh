#!/bin/bash

# Evaluate Rex-Omni model on three intention detection TRAIN datasets
# This script evaluates on COCO Outdoor, ScanNet, and EgoObject TRAIN sets
# Output format matches test evaluation for GRPO data filtering

set -e

# Configuration
MODEL_PATH="/home/hairong/hairong/code/IntentionDetection/finetuning/work_dirs/intention_sft_3epochs"
OUTPUT_FILE="evaluation_three_datasets_results_sft_3epochs_train_0111.json"
GPU=3  # GPU device ID
BACKEND="vllm"  # Use vllm for faster inference
# MAX_SAMPLES=100  # Set to a number like 100 to test on fewer samples, or leave empty for all

echo "========================================="
echo "Evaluating on Three TRAIN Datasets"
echo "========================================="
echo "Model: ${MODEL_PATH}"
echo "Backend: ${BACKEND}"
echo "GPU: ${GPU}"
if [ -n "$MAX_SAMPLES" ]; then
    echo "Max samples per dataset: ${MAX_SAMPLES}"
else
    echo "Using all training samples"
fi
echo ""
echo "This will evaluate on:"
echo "  1. COCO Outdoor TRAIN set"
echo "  2. ScanNet TRAIN set"
echo "  3. EgoObject TRAIN set"
echo "========================================="
echo ""

# Activate virtual environment
source ~/.pyenv/versions/rexomni/bin/activate

cd /home/hairong/hairong/code/IntentionDetection

# Set GPU
export CUDA_VISIBLE_DEVICES=${GPU}

# Run evaluation
if [ -n "$MAX_SAMPLES" ]; then
    python3 evaluate_intention_three_datasets_train.py \
        --checkpoint ${MODEL_PATH} \
        --output_file ${OUTPUT_FILE} \
        --backend ${BACKEND} \
        --max_samples ${MAX_SAMPLES}
else
    python3 evaluate_intention_three_datasets_train.py \
        --checkpoint ${MODEL_PATH} \
        --output_file ${OUTPUT_FILE} \
        --backend ${BACKEND}
fi

echo ""
echo "========================================="
echo "Evaluation Complete!"
echo "========================================="
echo "Results saved to: ${OUTPUT_FILE}"
echo ""
echo "This file contains:"
echo "  - Overall metrics (Mean IoU, AP@50, AP@75, AP@50:95)"
echo "  - Detailed predictions for each sample"
echo "  - IoU score for each sample (for GRPO filtering)"
echo ""
echo "Next steps:"
echo "  1. Analyze IoU distribution"
echo "  2. Select filtering threshold"
echo "  3. Filter training data for GRPO"
echo "========================================="


