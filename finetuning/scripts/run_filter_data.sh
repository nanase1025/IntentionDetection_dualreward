#!/bin/bash

# Step 1: Evaluate SFT model on all training samples and calculate IoU
# This script only does evaluation (slow, run once)
# After this, use apply_filter.sh to quickly filter with different thresholds

set -e

# Configuration
MODEL_PATH="/home/hairong/hairong/code/IntentionDetection/finetuning/work_dirs/intention_sft"
OUTPUT_DIR="finetuning/work_dirs/filtered_data"
GPU=3        # GPU device ID
BACKEND="vllm"  # Use vllm for faster inference (automatic batching)

echo "========================================="
echo "Evaluating SFT Model on Training Data"
echo "========================================="
echo "Model: ${MODEL_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "GPU: ${GPU}"
echo "Backend: ${BACKEND}"
echo ""
echo "This will evaluate ALL samples and save IoU scores."
echo "This takes ~2-3 hours but only needs to run ONCE."
echo "========================================="
echo ""

# Set CUDA device
export CUDA_VISIBLE_DEVICES=${GPU}

# Run evaluation using the existing evaluation script
cd /home/hairong/hairong/code/IntentionDetection

python3 evaluate_and_save_iou.py \
    --checkpoint ${MODEL_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --backend ${BACKEND}

echo ""
echo "========================================="
echo "Evaluation Complete!"
echo "========================================="
echo "IoU scores saved to: ${OUTPUT_DIR}"
echo ""
echo "Next steps:"
echo "1. Review the IoU distribution:"
echo "   python finetuning/scripts/analyze_iou_distribution.py"
echo ""
echo "2. Use apply_filter.sh to quickly filter with different thresholds:"
echo "   bash finetuning/scripts/apply_filter.sh 0.2 0.7  # Filter IoU in [0.2, 0.7]"
echo "   bash finetuning/scripts/apply_filter.sh 0.3 0.6  # Try different thresholds"
echo "========================================"
