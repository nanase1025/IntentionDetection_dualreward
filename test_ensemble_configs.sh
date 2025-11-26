#!/bin/bash

# Quick test script to compare different ensemble configurations
# Tests on a small subset (100 samples) to validate the hypothesis
export CUDA_VISIBLE_DEVICES=7
CHECKPOINT="/workspace/hairong/code/Rex-Omni/finetuning/work_dirs/ovigod_sft_5ep"
# MAX_SAMPLES=100
OUTPUT_DIR="ensemble_experiments"
BATCH_SIZE=32  # Batch size for inference acceleration
NUM_WORKERS=16  # Number of data loading workers

mkdir -p ${OUTPUT_DIR}

# echo "Testing different ensemble configurations on ${MAX_SAMPLES} samples..."
echo "This will take some time (each config needs n_samples * max_samples inferences)"
echo ""

# Baseline: temperature=0, no ensemble (like current evaluation)
echo "===================="
echo "Baseline (temp=0.0, n=1)"
echo "===================="
python evaluate_ovigod_ap.py \
    --checkpoint ${CHECKPOINT} \
    --backend vllm \
    --batch_size ${BATCH_SIZE} \
    --num_workers ${NUM_WORKERS} \
    --output_file ${OUTPUT_DIR}/grpo_sft_5ep_8rollouts_300steps_baseline.json || echo "Warning: Baseline failed, continuing..."

# Config 1: Small ensemble with moderate threshold
# echo ""
# echo "===================="
# echo "Config 1: temp=1.0, n=4, vote=0.5"
# echo "===================="
# python evaluate_ovigod_ap_ensemble.py \
#     --checkpoint ${CHECKPOINT} \
#     --n_samples 4 \
#     --vote_threshold 0.5 \
#     --iou_threshold 0.5\
#     --temperature 1.0 \
#     --backend vllm \
#     --batch_size ${BATCH_SIZE} \
#     --num_workers ${NUM_WORKERS} \
#     --output_file ${OUTPUT_DIR}/sft_5ep_n4_vote0.5_iou0.5_temp1.0.json

# Config 2: Medium ensemble with moderate threshold
# echo ""
# echo "===================="
# echo "Config 2: temp=1.0, n=5, vote=0.4"
# echo "===================="
# python evaluate_ovigod_ap_ensemble.py \
#     --checkpoint ${CHECKPOINT} \
#     --max_samples ${MAX_SAMPLES} \
#     --n_samples 5 \
#     --vote_threshold 0.4 \
#     --iou_threshold 0.5 \
#     --temperature 1.0 \
#     --backend vllm \
#     --batch_size ${BATCH_SIZE} \
#     --num_workers ${NUM_WORKERS} \
#     --output_file ${OUTPUT_DIR}/ensemble_n5_vote0.4.json

# Config 3: Medium ensemble with lower threshold
# echo ""
# echo "===================="
# echo "Config 3: temp=1.0, n=5, vote=0.3"
# echo "===================="
# python evaluate_ovigod_ap_ensemble.py \
#     --checkpoint ${CHECKPOINT} \
#     --max_samples ${MAX_SAMPLES} \
#     --n_samples 5 \
#     --vote_threshold 0.3 \
#     --iou_threshold 0.5 \
#     --temperature 1.0 \
#     --backend vllm \
#     --batch_size ${BATCH_SIZE} \
#     --num_workers ${NUM_WORKERS} \
#     --output_file ${OUTPUT_DIR}/ensemble_n5_vote0.3.json

# Config 4: Lower temperature
# echo ""
# echo "===================="
# echo "Config 4: temp=0.5, n=5, vote=0.4"
# echo "===================="
# python evaluate_ovigod_ap_ensemble.py \
#     --checkpoint ${CHECKPOINT} \
#     --max_samples ${MAX_SAMPLES} \
#     --n_samples 5 \
#     --vote_threshold 0.4 \
#     --iou_threshold 0.5 \
#     --temperature 0.5 \
#     --backend vllm \
#     --batch_size ${BATCH_SIZE} \
#     --num_workers ${NUM_WORKERS} \
#     --output_file ${OUTPUT_DIR}/ensemble_temp0.5_n5_vote0.4.json

# echo ""
# echo "===================="
# echo "All experiments completed!"
# echo "===================="
# echo ""
# echo "Comparing results..."

# Extract and compare AP@50 from all results
# python3 << 'EOF'
# import json
# import os
# from pathlib import Path

# output_dir = "ensemble_experiments"
# results = []

# for json_file in sorted(Path(output_dir).glob("*.json")):
#     with open(json_file) as f:
#         data = json.load(f)
    
#     config_name = json_file.stem
    
#     if "ensemble_config" in data:
#         # Ensemble result
#         config = data["ensemble_config"]
#         config_str = f"n={config['n_samples']}, vote={config['vote_threshold']}, temp={config['temperature']}"
#     else:
#         # Baseline result
#         config_str = "Baseline (temp=0, n=1)"
    
#     ap50 = data["metrics"]["AP@50"]
#     ap75 = data["metrics"]["AP@75"]
#     ap50_95 = data["metrics"]["AP@50:95"]
    
#     results.append({
#         'name': config_name,
#         'config': config_str,
#         'AP@50': ap50,
#         'AP@75': ap75,
#         'AP@50:95': ap50_95,
#     })

# # Print comparison table
# print("\n" + "="*100)
# print("RESULTS COMPARISON")
# print("="*100)
# print(f"{'Configuration':<40} {'AP@50':>12} {'AP@75':>12} {'AP@50:95':>12} {'vs Baseline':>15}")
# print("-"*100)

# baseline_ap50 = None
# for r in results:
#     if 'baseline' in r['name']:
#         baseline_ap50 = r['AP@50']
#         break

# for r in results:
#     ap50 = r['AP@50']
#     ap75 = r['AP@75']
#     ap50_95 = r['AP@50:95']
    
#     if baseline_ap50 and ap50 != baseline_ap50:
#         diff = (ap50 - baseline_ap50) / baseline_ap50 * 100
#         diff_str = f"{diff:+.2f}%"
#     else:
#         diff_str = "baseline"
    
#     print(f"{r['config']:<40} {ap50:>11.4f} {ap75:>11.4f} {ap50_95:>11.4f} {diff_str:>15}")

# print("="*100)
# print("\nBest configuration:")
# best = max(results, key=lambda x: x['AP@50:95'])
# print(f"  {best['config']}")
# print(f"  AP@50:95 = {best['AP@50:95']:.4f}")

# if baseline_ap50:
#     improvement = (best['AP@50:95'] - baseline_ap50) / baseline_ap50 * 100
#     print(f"  Improvement over baseline: {improvement:+.2f}%")

# EOF

# echo ""
# echo "Done! Check ${OUTPUT_DIR}/ for detailed results."

