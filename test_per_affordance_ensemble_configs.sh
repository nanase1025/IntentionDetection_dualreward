#!/bin/bash

# Test different ensemble configurations for Per-Affordance AP evaluation
# This script tests various ensemble parameters to find the optimal configuration

export CUDA_VISIBLE_DEVICES=7
CHECKPOINT="/workspace/hairong/code/Rex-Omni/finetuning/work_dirs/ovigod_sft_5ep"
OUTPUT_DIR="per_affordance_ensemble_results"
BATCH_SIZE=32
NUM_WORKERS=16

# Test on a subset for faster experimentation
# MAX_SAMPLES=50

mkdir -p ${OUTPUT_DIR}

echo "============================================================================"
echo "Testing Per-Affordance Ensemble Configurations"
echo "============================================================================"
echo "Checkpoint: ${CHECKPOINT}"
echo "Batch size: ${BATCH_SIZE}"
# echo "Max samples: ${MAX_SAMPLES}"
echo ""

# Configuration grid
# N_SAMPLES_LIST=(3 5 7)
N_SAMPLES_LIST=(4)
VOTE_THRESHOLD_LIST=(0.5)
# TEMPERATURE_LIST=(0.7 1.0)
TEMPERATURE_LIST=(1.0)
IOU_THRESHOLD=0.5

echo "Testing configurations:"
echo "  n_samples:       ${N_SAMPLES_LIST[@]}"
echo "  vote_threshold:  ${VOTE_THRESHOLD_LIST[@]}"
echo "  temperature:     ${TEMPERATURE_LIST[@]}"
echo "  iou_threshold:   ${IOU_THRESHOLD}"
echo ""

# Test each configuration
for n_samples in "${N_SAMPLES_LIST[@]}"; do
    for vote_threshold in "${VOTE_THRESHOLD_LIST[@]}"; do
        for temperature in "${TEMPERATURE_LIST[@]}"; do
            CONFIG_NAME="n${n_samples}_v${vote_threshold}_t${temperature}"
            OUTPUT_FILE="${OUTPUT_DIR}/${CONFIG_NAME}.json"
            
            echo "============================================================================"
            echo "Testing: n_samples=${n_samples}, vote_threshold=${vote_threshold}, temperature=${temperature}"
            echo "============================================================================"
            
            python3 evaluate_ovigod_ap_per_affordance_ensemble.py \
                --checkpoint ${CHECKPOINT} \
                --backend vllm \
                --n_samples ${n_samples} \
                --vote_threshold ${vote_threshold} \
                --temperature ${temperature} \
                --iou_threshold ${IOU_THRESHOLD} \
                --batch_size ${BATCH_SIZE} \
                --num_workers ${NUM_WORKERS} \
                --output_file ${OUTPUT_FILE}
            
            echo ""
            echo "Configuration completed: ${CONFIG_NAME}"
            echo ""
        done
    done
done

# Compare all results
echo "============================================================================"
echo "Comparing All Configurations"
echo "============================================================================"

python3 << 'EOF'
import json
import os
from pathlib import Path

output_dir = "per_affordance_ensemble_results"

# Collect all results
results = []
for json_file in sorted(Path(output_dir).glob("*.json")):
    try:
        with open(json_file) as f:
            data = json.load(f)
            
        config = data['ensemble_config']
        metrics = data['mAP_metrics']
        
        results.append({
            'config_name': json_file.stem,
            'n_samples': config['n_samples'],
            'vote_threshold': config['vote_threshold'],
            'temperature': config['temperature'],
            'iou_threshold': config['iou_threshold'],
            'mAP@50': metrics['mAP@50'],
            'mAP@75': metrics['mAP@75'],
            'mAP@50:95': metrics['mAP@50:95'],
        })
    except Exception as e:
        print(f"Error reading {json_file}: {e}")

if not results:
    print("No results found!")
    exit(1)

# Sort by mAP@50:95 (descending)
results.sort(key=lambda x: x['mAP@50:95'], reverse=True)

print("\n" + "="*120)
print("Configuration Comparison - Sorted by mAP@50:95")
print("="*120)
print(f"{'Rank':<6} {'Config':<20} {'N':<4} {'Vote':<6} {'Temp':<6} {'mAP@50':>10} {'mAP@75':>10} {'mAP@50:95':>10}")
print("-" * 120)

for rank, result in enumerate(results, 1):
    print(f"{rank:<6} {result['config_name']:<20} "
          f"{result['n_samples']:<4} {result['vote_threshold']:<6.2f} {result['temperature']:<6.2f} "
          f"{result['mAP@50']:>10.4f} {result['mAP@75']:>10.4f} {result['mAP@50:95']:>10.4f}")

# Best configuration
print("\n" + "="*120)
print("📊 Best Configuration")
print("="*120)
best = results[0]
print(f"Config:         {best['config_name']}")
print(f"n_samples:      {best['n_samples']}")
print(f"vote_threshold: {best['vote_threshold']}")
print(f"temperature:    {best['temperature']}")
print(f"mAP@50:         {best['mAP@50']:.4f} ({best['mAP@50']*100:.2f}%)")
print(f"mAP@75:         {best['mAP@75']:.4f} ({best['mAP@75']*100:.2f}%)")
print(f"mAP@50:95:      {best['mAP@50:95']:.4f} ({best['mAP@50:95']*100:.2f}%)")

# Analyze impact of each parameter
print("\n" + "="*120)
print("📈 Parameter Impact Analysis")
print("="*120)

# Group by n_samples
from collections import defaultdict

by_n_samples = defaultdict(list)
for r in results:
    by_n_samples[r['n_samples']].append(r['mAP@50:95'])

print("\nImpact of n_samples:")
for n in sorted(by_n_samples.keys()):
    avg = sum(by_n_samples[n]) / len(by_n_samples[n])
    print(f"  n_samples={n}: avg mAP@50:95 = {avg:.4f}")

# Group by vote_threshold
by_vote = defaultdict(list)
for r in results:
    by_vote[r['vote_threshold']].append(r['mAP@50:95'])

print("\nImpact of vote_threshold:")
for v in sorted(by_vote.keys()):
    avg = sum(by_vote[v]) / len(by_vote[v])
    print(f"  vote_threshold={v}: avg mAP@50:95 = {avg:.4f}")

# Group by temperature
by_temp = defaultdict(list)
for r in results:
    by_temp[r['temperature']].append(r['mAP@50:95'])

print("\nImpact of temperature:")
for t in sorted(by_temp.keys()):
    avg = sum(by_temp[t]) / len(by_temp[t])
    print(f"  temperature={t}: avg mAP@50:95 = {avg:.4f}")

print("\n" + "="*120)

EOF

echo ""
echo "All tests completed! Results saved in ${OUTPUT_DIR}/"

