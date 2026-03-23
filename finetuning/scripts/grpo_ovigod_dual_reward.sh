#!/bin/bash

# ========================================
# GRPO Training Script with Dual Reward (IoU + FG-CLIP) for OV-IGOD
# ========================================
# Dual Reward = IoU F1 (position accuracy) + FG-CLIP (semantic relevance)
# Full OV-IGOD train set, no data filtering
#
# Pipeline:
#   1. SFT on OV-IGOD train set (sft_ovigod.sh)
#   2. GRPO with Dual Reward on full train set (this script)
#
# Usage:
#   bash scripts/grpo_ovigod_dual_reward.sh

export OUTPUT_PATH="work_dirs/ovigod_grpo_dual_reward"
export EXP_NAME="ovigod_grpo_dual_reward"
export DEBUG_MODE="true"
export LOG_PATH="${OUTPUT_PATH}/log.txt"
export LOG_VISUALIZE_PATH="${OUTPUT_PATH}/visualizations"

# Dual Reward Configuration (FG-CLIP)
export DUAL_REWARD_ALPHA="0.5"           # IoU weight (position accuracy)
export DUAL_REWARD_BETA="0.5"            # FG-CLIP weight (semantic relevance)
export DUAL_REWARD_IOU_THRESHOLD="0.5"   # IoU threshold (kept for compatibility)
export DUAL_REWARD_CLIP_THRESHOLD="15.0" # FG-CLIP v1 threshold (~62% balanced accuracy)
export CLIP_MODEL_NAME="qihoo360/fg-clip-large"  # FG-CLIP v1 model
export FGCLIP_IMAGE_SIZE="336"           # FG-CLIP v1 fixed resolution
export FGCLIP_USE_LONG_TEXT="true"       # Long text mode (max_length=248)

# Data loading configuration
NUM_WORKERS=16

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

# =============================================
# SET THESE BEFORE RUNNING
# =============================================
# Path to SFT checkpoint (output of sft_ovigod.sh)
MODEL_PATH="work_dirs/ovigod_sft_3ep"

# GPU to use
export CUDA_VISIBLE_DEVICES=0

# Wandb settings
export WANDB_PROJECT="rex-omni-grpo-ovigod-dual"

echo "========================================="
echo "GRPO Training with Dual Reward (IoU + FG-CLIP) - OV-IGOD"
echo "========================================="
echo "Starting from SFT checkpoint: ${MODEL_PATH}"
echo "Output will be saved to: ${OUTPUT_PATH}"
echo "Debug logs: ${LOG_PATH}"
echo "Visualizations: ${LOG_VISUALIZE_PATH}"
echo ""
echo "Dual Reward Configuration:"
echo "  - IoU Weight (α): 0.5 - Position accuracy"
echo "  - FG-CLIP Weight (β): 0.5 - Semantic relevance"
echo "  - IoU: F1 score (continuous 0-1, no threshold)"
echo "  - FG-CLIP Threshold: 15.0 (v1 Long Text Mode, ~62% balanced accuracy)"
echo "  - FG-CLIP Model: qihoo360/fg-clip-large"
echo "  - Image Resolution: 336x336, Long Text: max_length=248"
echo ""
echo "Dataset: OV-IGOD full train set (6701 images, 15430 samples)"
echo "  - No data filtering, use all samples"
echo ""
echo "Strategy: IoU + FG-CLIP v1 Dual Reward"
echo "  IoU: F1 score (continuous 0-1, Precision + Recall)"
echo "  FG-CLIP v1: Binary reward (1.0 if score>15.0, else 0.0)"
echo "  Dual = α*IoU_F1 + β*CLIP_binary (α=0.5, β=0.5)"
echo "========================================="
echo ""

python3 -m verl.trainer.main \
    config=configs/grpo_ovigod.yaml \
    data.config_path="configs/grpo_ovigod_dual_reward.py" \
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
echo "Next steps:"
echo "1. Convert checkpoint: python convert_grpo_to_hf.py --checkpoint_path ${OUTPUT_PATH}/global_step_XXX/actor"
echo "2. Evaluate: python evaluate_ovigod_single_query.py --checkpoint ${OUTPUT_PATH}/global_step_XXX/actor/huggingface"
echo "3. Compare with SFT baseline and pure IoU GRPO"
echo ""
