#!/bin/bash

# GRPO training script for OV-IGOD affordance-based detection
# This script runs GRPO (reinforcement learning) on top of the SFT checkpoint

# export OUTPUT_PATH="work_dirs/ovigod_grpo"
export OUTPUT_PATH="work_dirs/ovigod_grpo_sft_5ep_8rollouts" # resume
# export EXP_NAME="ovigod_grpo"
export EXP_NAME="ovigod_grpo_sft_5ep_8rollouts" 
export DEBUG_MODE="true"  # Enable debug logging to monitor rewards
export LOG_PATH="${OUTPUT_PATH}/log.txt"
export LOG_VISUALIZE_PATH="${OUTPUT_PATH}/visualizations"  # Optional: save reward visualizations

# Data loading configuration
NUM_WORKERS=32  # Number of data loading workers (0=sequential, 2-8 for parallel loading)

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
MODEL_PATH="/workspace/hairong/code/Rex-Omni/finetuning/work_dirs/ovigod_sft_5ep"
# MODEL_PATH="work_dirs/ovigod_grpo/global_step_418/actor/huggingface" # resume
# Set CUDA device (use GPU 7 as you prefer)
export CUDA_VISIBLE_DEVICES=7

# Wandb settings
export WANDB_PROJECT="rex-omni-grpo-ovigod"

echo "Starting GRPO training from SFT checkpoint: ${MODEL_PATH}"
echo "Output will be saved to: ${OUTPUT_PATH}"
echo "Debug logs: ${LOG_PATH}"
# data.num_workers=${NUM_WORKERS} \
python3 -m verl.trainer.main \
    config=configs/grpo_ovigod.yaml \
    data.config_path="configs/grpo_ovigod.py" \
    data.num_workers=${NUM_WORKERS} \
    data.format_prompt="verl/configs/r1v_format.jinja" \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=${EXP_NAME} \
    trainer.n_gpus_per_node=1 \
    worker.actor.global_batch_size=16 \
    data.rollout_batch_size=16 \
    worker.actor.micro_batch_size_per_device_for_update=2 \
    worker.actor.micro_batch_size_per_device_for_experience=4 \
    worker.rollout.n=8 \
    worker.rollout.temperature=1.0 \
    trainer.total_epochs=1 \
    trainer.save_checkpoint_path=${OUTPUT_PATH} \
    trainer.save_freq=100 \
    trainer.save_limit=10

echo "GRPO training completed!"
echo "Checkpoints saved in: ${OUTPUT_PATH}"
echo "Review reward logs in: ${LOG_PATH}"

