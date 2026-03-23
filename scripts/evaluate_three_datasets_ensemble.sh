#!/bin/bash

# Evaluate Rex-Omni model on three intention detection datasets with Test-Time Ensemble
# This script uses temperature sampling + voting for improved accuracy

set -e

# Configuration
MODEL_PATH="/home/hairong/hairong/code/IntentionDetection/finetuning/work_dirs/intention_grpo_iou_lt_06/global_step_828/actor/huggingface"
OUTPUT_FILE="evaluation_three_datasets_results_grpo_iou_lt_06_0112_ensemble.json"
GPU=2  # GPU device ID
BACKEND="vllm"  # Use vllm for faster inference

# Ensemble parameters
N_SAMPLES=4          # Number of predictions per image (more = better but slower)
VOTE_THRESHOLD=0.4   # Minimum vote ratio to keep a box (40% of samples must agree)
IOU_THRESHOLD=0.5    # IoU threshold for clustering similar boxes
TEMPERATURE=1.0      # Sampling temperature (1.0 matches GRPO training)
BATCH_SIZE=1         # Batch size (total inferences = batch_size * n_samples)

# Testing parameters
# MAX_SAMPLES=20  # Set to a number like 100 to test on fewer samples, or leave empty for all

echo "========================================="
echo "Evaluating on Three Datasets with Ensemble"
echo "========================================="
echo "Model: ${MODEL_PATH}"
echo "Backend: ${BACKEND}"
echo "GPU: ${GPU}"
echo ""
echo "Ensemble Configuration:"
echo "  n_samples:       ${N_SAMPLES}"
echo "  vote_threshold:  ${VOTE_THRESHOLD} (${VOTE_THRESHOLD}x100% of samples must agree)"
echo "  iou_threshold:   ${IOU_THRESHOLD}"
echo "  temperature:     ${TEMPERATURE}"
echo "  batch_size:      ${BATCH_SIZE}"
echo ""
if [ -n "$MAX_SAMPLES" ]; then
    echo "Max samples per dataset: ${MAX_SAMPLES}"
else
    echo "Using all test samples"
fi
echo ""
echo "Output: ${OUTPUT_FILE}"
echo "========================================="
echo ""

# Activate virtual environment
source ~/.pyenv/versions/rexomni/bin/activate

# Set GPU
export CUDA_VISIBLE_DEVICES=${GPU}

# Change to project directory
cd /home/hairong/hairong/code/IntentionDetection

# Build command
CMD="python3 evaluate_intention_three_datasets_ensemble.py \
    --checkpoint ${MODEL_PATH} \
    --backend ${BACKEND} \
    --n_samples ${N_SAMPLES} \
    --vote_threshold ${VOTE_THRESHOLD} \
    --iou_threshold ${IOU_THRESHOLD} \
    --temperature ${TEMPERATURE} \
    --batch_size ${BATCH_SIZE} \
    --output_file ${OUTPUT_FILE}"

# Add max_samples if specified
if [ -n "$MAX_SAMPLES" ]; then
    CMD="${CMD} --max_samples ${MAX_SAMPLES}"
fi

# Run evaluation
echo "Running ensemble evaluation..."
echo ""
eval ${CMD}

echo ""
echo "========================================="
echo "Evaluation Complete!"
echo "========================================="
echo "Results saved to: ${OUTPUT_FILE}"
echo ""
echo "To view results:"
echo "  cat ${OUTPUT_FILE} | jq '.per_dataset_metrics'"
echo ""
echo "Tips for tuning ensemble parameters:"
echo "  - Increase n_samples (e.g., 10) for better accuracy (but slower)"
echo "  - Decrease vote_threshold (e.g., 0.3) to get more boxes"
echo "  - Increase vote_threshold (e.g., 0.5) for higher precision"
echo "  - Use temperature=1.0 to match GRPO training distribution"
echo "========================================="

