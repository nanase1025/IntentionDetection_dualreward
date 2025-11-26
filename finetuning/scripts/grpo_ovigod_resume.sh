#!/bin/bash

# GRPO Resume Training Script for OV-IGOD
# Resume from the latest checkpoint

export OUTPUT_PATH="work_dirs/ovigod_grpo"
export EXP_NAME="ovigod_grpo_resumed"
export DEBUG_MODE="true"
export LOG_PATH="${OUTPUT_PATH}/log_resumed.txt"
export LOG_VISUALIZE_PATH="${OUTPUT_PATH}/visualizations"

# ⭐ Checkpoint to resume from
CHECKPOINT_TO_RESUME="work_dirs/ovigod_grpo/global_step_418"

# Check if checkpoint exists
if [ ! -d "${CHECKPOINT_TO_RESUME}" ]; then
    echo "Error: Checkpoint not found at ${CHECKPOINT_TO_RESUME}"
    echo "Available checkpoints:"
    ls -d ${OUTPUT_PATH}/global_step_* 2>/dev/null || echo "  None found"
    exit 1
fi

echo "Resuming GRPO training from: ${CHECKPOINT_TO_RESUME}"

# ⭐ Clean up any existing Ray/vLLM processes first
echo "Cleaning up old processes..."
pkill -9 -f "ray::" 2>/dev/null || true
pkill -9 -f "verl.trainer.main" 2>/dev/null || true
pkill -9 -f "vllm" 2>/dev/null || true
rm -rf /tmp/ray/* 2>/dev/null || true
sleep 2
echo "Cleanup complete."

# Create output directories
mkdir -p ${OUTPUT_PATH}
mkdir -p ${LOG_VISUALIZE_PATH}

set -x

export PYTHONUNBUFFERED=1

# Disable Ray cluster discovery
export RAY_DISABLE_IMPORT_WARNING=1
export RAY_ADDRESS=""
export RAY_CLIENT_MODE=""

# Force VLLM to use local Ray cluster
export VLLM_USE_RAY_COMPILED_DAG=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Clear any Ray temp files
rm -rf /tmp/ray/* 2>/dev/null || true

# Your best SFT checkpoint path (will be overridden by loaded checkpoint)
MODEL_PATH="work_dirs/ovigod_sft/checkpoint-627"

# Set CUDA device
export CUDA_VISIBLE_DEVICES=7

# Wandb settings
export WANDB_PROJECT="rex-omni-grpo-ovigod"

echo "Starting GRPO training from checkpoint: ${CHECKPOINT_TO_RESUME}"
echo "Output will be saved to: ${OUTPUT_PATH}"
echo "Debug logs: ${LOG_PATH}"

python3 -m verl.trainer.main \
    config=configs/grpo_ovigod.yaml \
    data.config_path="configs/grpo_ovigod.py" \
    data.format_prompt="verl/configs/r1v_format.jinja" \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=${EXP_NAME} \
    trainer.n_gpus_per_node=1 \
    trainer.load_checkpoint_path=${CHECKPOINT_TO_RESUME} \
    trainer.total_epochs=2 \
    worker.actor.global_batch_size=16 \
    data.rollout_batch_size=16 \
    worker.actor.micro_batch_size_per_device_for_update=2 \
    worker.actor.micro_batch_size_per_device_for_experience=4 \
    worker.rollout.n=4 \
    trainer.save_checkpoint_path=${OUTPUT_PATH} \
    trainer.save_freq=50 \
    trainer.save_limit=3

echo "GRPO resumed training completed!"
echo "Checkpoints saved in: ${OUTPUT_PATH}"
echo "Review reward logs in: ${LOG_PATH}"

