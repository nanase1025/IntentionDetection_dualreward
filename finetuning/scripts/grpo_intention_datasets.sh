#!/bin/bash

# ========================================
# GRPO Training Script for Intention Datasets
# Three Datasets: COCO Outdoor, ScanNet, EgoObject
# ========================================
# This script runs GRPO (reinforcement learning) on top of the SFT checkpoint

export OUTPUT_PATH="work_dirs/intention_grpo_new"
export EXP_NAME="intention_datasets_grpo"
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

# Your best SFT checkpoint path (最后一个checkpoint)
MODEL_PATH="/home/hairong/hairong/code/IntentionDetection/finetuning/work_dirs/intention_sft_3epochs"

# Set CUDA device (use GPU 2)
export CUDA_VISIBLE_DEVICES=2

# Wandb settings
export WANDB_PROJECT="rex-omni-grpo-intention"
export WANDB_API_KEY="d0891adc2fc5fb80fce98ca48404b2dca194cd8c"

echo "========================================="
echo "GRPO Training for Intention Datasets"
echo "========================================="
echo "Starting from SFT checkpoint: ${MODEL_PATH}"
echo "Output will be saved to: ${OUTPUT_PATH}"
echo "Debug logs: ${LOG_PATH}"
echo "Visualizations: ${LOG_VISUALIZE_PATH}"
echo ""
echo "Datasets:"
echo "  - COCO Outdoor: 8,000 samples"
echo "  - ScanNet: 7,383 samples"
echo "  - EgoObject: 11,528 samples"
echo "  Total: 26,911 samples"
echo "========================================="
echo ""

python3 -m verl.trainer.main \
    config=configs/grpo_intention_datasets.yaml \
    data.config_path="configs/grpo_intention_datasets.py" \
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
    trainer.save_freq=100 \
    trainer.save_limit=5

echo ""
echo "========================================="
echo "GRPO training completed!"
echo "========================================="
echo "Checkpoints saved in: ${OUTPUT_PATH}"
echo "Review reward logs in: ${LOG_PATH}"
echo "Visualizations in: ${LOG_VISUALIZE_PATH}"
echo ""

