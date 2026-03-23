#!/bin/bash

# ========================================
# GRPO Training Script with Dual Reward (IoU + CLIP)
# Using samples with IoU < 0.8
# ========================================
# This script runs GRPO with combined IoU and CLIP rewards
# IoU ensures position accuracy, CLIP ensures semantic understanding

export OUTPUT_PATH="work_dirs/intention_grpo_dual_reward_iou_lt_08"
export EXP_NAME="intention_grpo_dual_reward_iou_lt_08"
export DEBUG_MODE="true"  # Enable debug logging to monitor rewards
export LOG_PATH="${OUTPUT_PATH}/log.txt"
export LOG_VISUALIZE_PATH="${OUTPUT_PATH}/visualizations"

# Dual Reward Configuration
export DUAL_REWARD_ALPHA="0.5"          # IoU weight (position accuracy)
export DUAL_REWARD_BETA="0.5"           # CLIP weight (semantic relevance)
export DUAL_REWARD_IOU_THRESHOLD="0.5"  # IoU threshold
export DUAL_REWARD_CLIP_THRESHOLD="15.0" # FG-CLIP v1 threshold (tested on COCO, ~62% balanced accuracy)
export CLIP_MODEL_NAME="qihoo360/fg-clip-large"  # FG-CLIP v1 model
export FGCLIP_IMAGE_SIZE="336"          # FG-CLIP v1 fixed resolution
export FGCLIP_USE_LONG_TEXT="true"      # Use long text mode (max_length=248)

# Data loading configuration
NUM_WORKERS=16  # Number of data loading workers

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
MODEL_PATH="/home/hairong/hairong/code/IntentionDetection/finetuning/work_dirs/intention_sft_3epochs"

# Set CUDA device (use GPU 2)
export CUDA_VISIBLE_DEVICES=2

# Wandb settings
export WANDB_PROJECT="rex-omni-grpo-intention-dual"
export WANDB_API_KEY="d0891adc2fc5fb80fce98ca48404b2dca194cd8c"

echo "========================================="
echo "GRPO Training with Dual Reward (IoU + FG-CLIP)"
echo "========================================="
echo "Starting from SFT checkpoint: ${MODEL_PATH}"
echo "Output will be saved to: ${OUTPUT_PATH}"
echo "Debug logs: ${LOG_PATH}"
echo "Visualizations: ${LOG_VISUALIZE_PATH}"
echo ""
echo "Dual Reward Configuration:"
echo "  - IoU Weight (α): 0.5 - Position accuracy"
echo "  - FG-CLIP Weight (β): 0.5 - Semantic relevance"
echo "  - IoU Threshold: 0.5"
echo "  - FG-CLIP Threshold: 15.0 (v1 Long Text Mode, ~62% balanced accuracy)"
echo "  - FG-CLIP Model: qihoo360/fg-clip-large"
echo "  - Image Resolution: 336x336, Long Text: max_length=248"
echo ""
echo "Training Datasets (IoU < 0.8 - Moderately Difficult):"
echo "  - COCO Outdoor: 2,342 samples (29.28% of original)"
echo "  - ScanNet: 3,954 samples (53.56% of original)"
echo "  - EgoObject: 2,759 samples (23.93% of original)"
echo "  Total: 9,055 samples (33.65% of 26,911)"
echo ""
echo "Strategy: IoU + FG-CLIP v1 Dual Reward"
echo "  ✅ IoU: F1 score (continuous 0-1, Precision + Recall)"
echo "  ✅ FG-CLIP v1: Binary reward (1.0 if score>15.0, else 0.0)"
echo "  ✅ Dual = α*IoU_F1 + β*CLIP_binary (α=0.5, β=0.5)"
echo "========================================="
echo ""

python3 -m verl.trainer.main \
    config=configs/grpo_intention_datasets.yaml \
    data.config_path="configs/sft_intention_datasets_grpo_dual_reward_iou_lt_08.py" \
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
    worker.rollout.gpu_memory_utilization=0.6 \
    trainer.total_epochs=1 \
    trainer.save_checkpoint_path=${OUTPUT_PATH} \
    trainer.save_freq=100 \
    trainer.save_limit=5

echo ""
echo "========================================="
echo "GRPO training with Dual Reward (IoU + FG-CLIP) completed!"
echo "========================================="
echo "Checkpoints saved in: ${OUTPUT_PATH}"
echo "Review reward logs in: ${LOG_PATH}"
echo "Visualizations in: ${LOG_VISUALIZE_PATH}"
echo ""
echo "Check the logs for:"
echo "  - IoU Reward: F1 score (continuous 0.0-1.0, P+R based)"
echo "  - FG-CLIP Reward: Binary (1.0 if score>15.0, else 0.0)"
echo "  - Dual Reward: α*IoU_F1 + β*CLIP_binary"
echo "  - Expected range: 0.0 to 1.0 (with α=β=0.5)"
echo ""
echo "Newo yao we
echo "1. Evaluate the GRPO checkpoint on test sets"
echo "2. Compare with:"
echo "   - SFT baseline (intention_sft_3epochs)"
echo "   - Pure IoU GRPO (intention_grpo_iou_lt_08)"
echo "   - Dual Reward IoU < 0.6 (intention_grpo_dual_reward)"
echo "3. Check if Dual Reward improved both location and semantic understanding"
echo ""
