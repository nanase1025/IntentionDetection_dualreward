#!/bin/bash

# ========================================
# GRPO Training Script for Filtered Intention Datasets
# Using FILTERED data with IoU in [0.2, 0.8] - Medium difficulty samples
# Total: 4,450 samples (COCO: 1,516, ScanNet: 1,834, EgoObject: 1,100)
# ========================================
# This script runs GRPO (reinforcement learning) on top of the SFT checkpoint

export OUTPUT_PATH="work_dirs/intention_grpo_filtered_epoch1"
export EXP_NAME="intention_filtered_iou_0.2_0.8"
export DEBUG_MODE="true"  # Enable debug logging to monitor rewards
export LOG_PATH="${OUTPUT_PATH}/log.txt"
export LOG_VISUALIZE_PATH="${OUTPUT_PATH}/visualizations"  # Optional: save reward visualizations

# Data loading configuration
NUM_WORKERS=32  # Number of data loading workers

# Create output directories
mkdir -p ${OUTPUT_PATH}
mkdir -p ${LOG_VISUALIZE_PATH}

set -x

export PYTHONUNBUFFERED=1

# Disable Ray cluster discovery to force local cluster
export RAY_DISABLE_IMPORT_WARNING=1
export RAY_ADDRESS=""
export RAY_CLIENT_MODE=""

# Force VLLM to use local Ray cluster
export VLLM_USE_RAY_COMPILED_DAG=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Clear any Ray temp files
rm -rf /tmp/ray/* 2>/dev/null || true

# Your best SFT checkpoint path
MODEL_PATH="/home/hairong/hairong/code/IntentionDetection/finetuning/work_dirs/intention_sft"

# Set CUDA device (use GPU 3)
export CUDA_VISIBLE_DEVICES=3

# Wandb settings
export WANDB_PROJECT="rex-omni-grpo-intention-filtered"
export WANDB_API_KEY="d0891adc2fc5fb80fce98ca48404b2dca194cd8c"

echo "========================================="
echo "GRPO Training for FILTERED Intention Datasets"
echo "========================================="
echo "Starting from SFT checkpoint: ${MODEL_PATH}"
echo "Output will be saved to: ${OUTPUT_PATH}"
echo "Debug logs: ${LOG_PATH}"
echo "Visualizations: ${LOG_VISUALIZE_PATH}"
echo ""
echo "Filtered Datasets (IoU: 0.2-0.8):"
echo "  - COCO Outdoor: 1,516 samples (18.9% of original)"
echo "  - ScanNet: 1,834 samples (24.8% of original)"
echo "  - EgoObject: 1,100 samples (9.5% of original)"
echo "  Total: 4,450 samples (16.5% of original)"
echo ""
echo "Strategy: Focus on medium-difficulty samples"
echo "  - Excluded IoU < 0.2: Model didn't understand"
echo "  - Excluded IoU > 0.8: Already very good"
echo "  - Kept IoU 0.2-0.8: Most potential for improvement"
echo "========================================="
echo ""

python3 -m verl.trainer.main \
    config=configs/grpo_intention_datasets_filtered.yaml \
    data.config_path="configs/grpo_intention_datasets_filtered.py" \
    data.num_workers=${NUM_WORKERS} \
    data.format_prompt="verl/configs/r1v_format.jinja" \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=${EXP_NAME} \
    trainer.n_gpus_per_node=1 \
    worker.actor.global_batch_size=8 \
    data.rollout_batch_size=8 \
    worker.actor.micro_batch_size_per_device_for_update=2 \
    worker.actor.micro_batch_size_per_device_for_experience=4 \
    worker.rollout.n=4 \
    worker.rollout.temperature=1.0 \
    worker.rollout.gpu_memory_utilization=0.65 \
    trainer.total_epochs=1 \
    trainer.save_checkpoint_path=${OUTPUT_PATH} \
    trainer.save_freq=50 \
    trainer.save_limit=5

echo ""
echo "========================================="
echo "GRPO training completed!"
echo "========================================="
echo "Checkpoints saved in: ${OUTPUT_PATH}"
echo "Review reward logs in: ${LOG_PATH}"
echo "Visualizations in: ${LOG_VISUALIZE_PATH}"
echo ""
echo "Next steps:"
echo "1. Evaluate the GRPO checkpoint on test sets"
echo "2. Compare with SFT baseline to measure improvement"
echo "3. Try different IoU ranges if needed (e.g., 0.3-0.7)"
echo ""

