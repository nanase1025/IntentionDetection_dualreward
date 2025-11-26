#!/bin/bash

NNODES=1
GPUS_PER_NODE=1  # 使用单张GPU

export MASTER_ADDR=localhost
export MASTER_PORT=29500
export WANDB_PROJECT="rex-omni-ovigod"  # 自定义 wandb project 名称

run_name="rexomni_ovigod_affordance-5ep"

torchrun \
    --nnodes=${NNODES} \
    --nproc_per_node=${GPUS_PER_NODE} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} train.py \
    --config configs/sft_ovigod.py \
    --deepspeed scripts/zero2.json \
    --data_flatten False \
    --tune_mm_vision True \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 \
    --output_dir work_dirs/ovigod_sft_5ep \
    --num_train_epochs 5 \
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

