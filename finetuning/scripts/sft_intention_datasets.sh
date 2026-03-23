#!/bin/bash

# ========================================
# Rex-Omni SFT Training Script
# Three Intention Datasets
# ========================================
export CUDA_VISIBLE_DEVICES=3
NNODES=1
GPUS_PER_NODE=1  # 根据实际GPU数量调整

export MASTER_ADDR=localhost
export MASTER_PORT=29500

# WandB 配置
export WANDB_API_KEY="d0891adc2fc5fb80fce98ca48404b2dca194cd8c"
export WANDB_PROJECT="rex-omni-intention"  # 项目名称

run_name="rexomni_intention_sft"

torchrun \
    --nnodes=${NNODES} \
    --nproc_per_node=${GPUS_PER_NODE} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} train.py \
    --config configs/sft_intention_datasets.py \
    --deepspeed scripts/zero2.json \
    --data_flatten False \
    --tune_mm_vision True \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 \
    --output_dir work_dirs/intention_sft_3epochs \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 5 \
    --learning_rate 2e-5 \
    --mm_projector_lr 2e-5 \
    --vision_tower_lr 2e-6 \
    --optim adamw_torch \
    --warmup_ratio 0.03 \
    --weight_decay 0.01 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --run_name ${run_name} \
    --report_to wandb

