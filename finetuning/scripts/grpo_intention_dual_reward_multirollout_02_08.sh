#!/bin/bash

# ========================================
# GRPO Training Script with Dual Reward - Multi-Rollout Filtered (0.2-0.8)
# ========================================
# This script runs GRPO with combined IoU and CLIP rewards
# on samples filtered by multi-rollout evaluation (0.2 <= mean_iou < 0.8)
#
# IoU ensures position accuracy, CLIP ensures semantic understanding

export OUTPUT_PATH="work_dirs/intention_grpo_dual_reward_multirollout_02_08_8rollouts"
export EXP_NAME="intention_grpo_dual_reward_multirollout_02_08_8rollouts"
export DEBUG_MODE="true"  # Enable debug logging to monitor rewards
export LOG_PATH="${OUTPUT_PATH}/log.txt"
export LOG_VISUALIZE_PATH="${OUTPUT_PATH}/visualizations"

# Dual Reward Configuration
export DUAL_REWARD_ALPHA="0.5"          # IoU weight (position accuracy)
export DUAL_REWARD_BETA="0.5"           # CLIP weight (semantic relevance)
export DUAL_REWARD_IOU_THRESHOLD="0.5"  # IoU threshold
export DUAL_REWARD_CLIP_THRESHOLD="0.22" # CLIP cosine similarity threshold (empirically tuned)
export CLIP_MODEL_NAME="openai/clip-vit-base-patch32"  # CLIP model

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
echo "GRPO Training with Dual Reward - Multi-Rollout Filtered"
echo "========================================="
echo "Starting from SFT checkpoint: ${MODEL_PATH}"
echo "Output will be saved to: ${OUTPUT_PATH}"
echo "Debug logs: ${LOG_PATH}"
echo "Visualizations: ${LOG_VISUALIZE_PATH}"
echo ""
echo "Dual Reward Configuration:"
echo "  - IoU Weight (α): 0.5 - Position accuracy"
echo "  - CLIP Weight (β): 0.5 - Semantic relevance"
echo "  - IoU Threshold: 0.5"
echo "  - CLIP Threshold: 0.0 (cosine similarity: >0 = similar, <0 = dissimilar)"
echo ""
echo "Training Datasets (0.2 <= mean_iou < 0.8):"
echo "  - COCO Outdoor: 1,473 samples (18.41% of 8,000)"
echo "  - ScanNet: 1,919 samples (25.99% of 7,383)"
echo "  - EgoObject: 1,112 samples (9.65% of 11,528)"
echo "  Total: 4,504 samples (16.74% of 26,911)"
echo ""
echo "Filtering Strategy:"
echo "  ✅ Excluded: mean_iou < 0.2 (completely failed)"
echo "  ✅ Included: 0.2 <= mean_iou < 0.8 (improvement potential)"
echo "  ✅ Excluded: mean_iou >= 0.8 (already excellent)"
echo ""
echo "Dual Reward Strategy:"
echo "  ✅ IoU ensures location accuracy"
echo "  ✅ CLIP ensures semantic understanding"
echo "  ✅ Both must be correct for high reward"
echo ""
echo "Advantages:"
echo "  - Model already has some understanding (not random)"
echo "  - Clear room for improvement (not perfect)"
echo "  - Dual reward guides both geometric and semantic learning"
echo "========================================="
echo ""

python3 -m verl.trainer.main \
    config=configs/grpo_intention_datasets.yaml \
    data.config_path="configs/sft_intention_datasets_grpo_dual_reward_multirollout_02_08.py" \
    data.num_workers=${NUM_WORKERS} \
    data.format_prompt="verl/configs/r1v_format.jinja" \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=${EXP_NAME} \
    trainer.n_gpus_per_node=1 \
    worker.actor.global_batch_size=8 \
    data.rollout_batch_size=8 \
    worker.actor.micro_batch_size_per_device_for_update=2 \
    worker.actor.micro_batch_size_per_device_for_experience=4 \
    worker.rollout.n=8 \
    worker.rollout.temperature=1.0 \
    trainer.total_epochs=1 \
    trainer.save_checkpoint_path=${OUTPUT_PATH} \
    trainer.save_freq=100 \
    trainer.save_limit=1

echo ""
echo "========================================="
echo "GRPO training with Dual Reward completed!"
echo "========================================="
echo "Checkpoints saved in: ${OUTPUT_PATH}"
echo "Review reward logs in: ${LOG_PATH}"
echo "Visualizations in: ${LOG_VISUALIZE_PATH}"
echo ""
echo "Check the logs for:"
echo "  - Average IoU Reward (should be 0.3-0.6)"
echo "  - Average CLIP Reward (should be 0.3-0.6)"
echo "  - Average Dual Reward (should be 0.3-0.6)"
echo ""
echo "Next steps:"
echo "1. Evaluate the GRPO checkpoint on test sets"
echo "2. Compare with:"
echo "   - SFT baseline (intention_sft_3epochs)"
echo "   - Pure IoU GRPO (intention_grpo_multirollout_02_08)"
echo "   - Other Dual Reward variants (IoU < 0.6, IoU < 0.8)"
echo "3. Analyze if dual reward improved both location and semantic understanding"
echo ""
