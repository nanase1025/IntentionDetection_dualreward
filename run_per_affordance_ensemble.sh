#!/bin/bash

# Quick test script for Per-Affordance AP with Ensemble
# For testing a single configuration quickly

export CUDA_VISIBLE_DEVICES=7
CHECKPOINT="/workspace/hairong/code/Rex-Omni/finetuning/work_dirs/ovigod_sft_5ep"
OUTPUT_FILE="per_affordance_ensemble_quick_test.json"

# Ensemble configuration
N_SAMPLES=5
VOTE_THRESHOLD=0.4
TEMPERATURE=1.0
IOU_THRESHOLD=0.5

# Inference configuration
BATCH_SIZE=16
NUM_WORKERS=8
MAX_SAMPLES=10  # Quick test on 10 samples

echo "============================================================================"
echo "Per-Affordance AP Evaluation with Ensemble"
echo "============================================================================"
echo "Checkpoint:      ${CHECKPOINT}"
echo ""
echo "Ensemble Config:"
echo "  n_samples:       ${N_SAMPLES}"
echo "  vote_threshold:  ${VOTE_THRESHOLD}"
echo "  temperature:     ${TEMPERATURE}"
echo "  iou_threshold:   ${IOU_THRESHOLD}"
echo ""
echo "Inference Config:"
echo "  batch_size:      ${BATCH_SIZE}"
echo "  num_workers:     ${NUM_WORKERS}"
echo "  max_samples:     ${MAX_SAMPLES}"
echo ""

python3 evaluate_ovigod_ap_per_affordance_ensemble.py \
    --checkpoint ${CHECKPOINT} \
    --backend vllm \
    --max_samples ${MAX_SAMPLES} \
    --n_samples ${N_SAMPLES} \
    --vote_threshold ${VOTE_THRESHOLD} \
    --temperature ${TEMPERATURE} \
    --iou_threshold ${IOU_THRESHOLD} \
    --batch_size ${BATCH_SIZE} \
    --num_workers ${NUM_WORKERS} \
    --output_file ${OUTPUT_FILE}

echo ""
echo "Evaluation completed! Results saved to: ${OUTPUT_FILE}"

