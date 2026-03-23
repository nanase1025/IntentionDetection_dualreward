#!/bin/bash

# Evaluate Rex-Omni model on three intention detection datasets
# This script evaluates on COCO Outdoor, ScanNet, and EgoObject test sets

set -e

# Configuration
MODEL_PATH="/home/hairong/hairong/code/IntentionDetection_dualreward/finetuning/work_dirs/intention_grpo_iou_lt_06/global_step_828/actor/huggingface"
# OUTPUT_FILE="evaluation_three_datasets_results_grpo_iou_lt_06_0112.json"
OUTPUT_FILE="evaluation_three_datasets_results_grpo_iou_lt_06_0130_realdual.json"
GPU=2  # GPU device ID
BACKEND="vllm"  # Use vllm for faster inference
# MAX_SAMPLES="100"  # Set to a number like 100 to test on fewer samples, or leave empty for all
# MAX_SAMPLES=100  # Set to a number like 100 to test on fewer samples, or leave empty for all

echo "========================================="
echo "Evaluating on Three Datasets"
echo "========================================="
echo "Model: ${MODEL_PATH}"
echo "Backend: ${BACKEND}"
echo "GPU: ${GPU}"
if [ -n "$MAX_SAMPLES" ]; then
    echo "Max samples per dataset: ${MAX_SAMPLES}"
else
    echo "Using all samples"
fi
echo ""
echo "This will evaluate on:"
echo "  1. COCO Outdoor test set"
echo "  2. ScanNet test set"
echo "  3. EgoObject test set"
echo "========================================="
echo ""

# Activate virtual environment
source ~/.pyenv/versions/rexomni/bin/activate

cd /home/hairong/hairong/code/IntentionDetection

# Set GPU
export CUDA_VISIBLE_DEVICES=${GPU}

# Run evaluation
if [ -n "$MAX_SAMPLES" ]; then
    python3 evaluate_intention_three_datasets.py \
        --checkpoint ${MODEL_PATH} \
        --output_file ${OUTPUT_FILE} \
        --backend ${BACKEND} \
        --max_samples ${MAX_SAMPLES}
else
    python3 evaluate_intention_three_datasets.py \
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
echo "To view results:"
echo "  python3 -c 'import json; print(json.dumps(json.load(open(\"${OUTPUT_FILE}\")), indent=2))'"
echo "========================================="

# #!/bin/bash

# # Evaluate Rex-Omni model on three intention detection datasets
# # This script evaluates on COCO Outdoor, ScanNet, and EgoObject test sets

# set -e

# # Configuration
# MODEL_PATH="/home/hairong/hairong/code/IntentionDetection/finetuning/work_dirs/intention_sft_3epochs"
# OUTPUT_FILE="evaluation_three_datasets_results_sft_3epochs_test_0112.json"
# GPU=2  # GPU device ID
# BACKEND="vllm"  # Use vllm for faster inference
# # MAX_SAMPLES="100"  # Set to a number like 100 to test on fewer samples, or leave empty for all

# echo "========================================="
# echo "Evaluating on Three Datasets"
# echo "========================================="
# echo "Model: ${MODEL_PATH}"
# echo "Backend: ${BACKEND}"
# echo "GPU: ${GPU}"
# if [ -n "$MAX_SAMPLES" ]; then
#     echo "Max samples per dataset: ${MAX_SAMPLES}"
# else
#     echo "Using all samples"
# fi
# echo ""
# echo "This will evaluate on:"
# echo "  1. COCO Outdoor test set"
# echo "  2. ScanNet test set"
# echo "  3. EgoObject test set"
# echo "========================================="
# echo ""

# # Activate virtual environment
# source ~/.pyenv/versions/rexomni/bin/activate

# cd /home/hairong/hairong/code/IntentionDetection

# # Set GPU
# export CUDA_VISIBLE_DEVICES=${GPU}

# # Run evaluation
# if [ -n "$MAX_SAMPLES" ]; then
#     python3 evaluate_intention_three_datasets.py \
#         --checkpoint ${MODEL_PATH} \
#         --output_file ${OUTPUT_FILE} \
#         --backend ${BACKEND} \
#         --max_samples ${MAX_SAMPLES}
# else
#     python3 evaluate_intention_three_datasets.py \
#         --checkpoint ${MODEL_PATH} \
#         --output_file ${OUTPUT_FILE} \
#         --backend ${BACKEND}
# fi

# echo ""
# echo "========================================="
# echo "Evaluation Complete!"
# echo "========================================="
# echo "Results saved to: ${OUTPUT_FILE}"
# echo ""
# echo "To view results:"
# echo "  python3 -c 'import json; print(json.dumps(json.load(open(\"${OUTPUT_FILE}\")), indent=2))'"
# echo "========================================="
