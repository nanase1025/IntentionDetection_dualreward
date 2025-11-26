#!/bin/bash

# Compare AP evaluation methods: Class-Agnostic vs Per-Affordance
# This script runs both evaluation methods and compares the results

export CUDA_VISIBLE_DEVICES=7
CHECKPOINT="/workspace/hairong/code/Rex-Omni/finetuning/work_dirs/ovigod_grpo/global_step_418/actor/huggingface"
# CHECKPOINT="IDEA-Research/Rex-Omni"
OUTPUT_DIR="ap_method_comparison"
BATCH_SIZE=64
NUM_WORKERS=16

# Test on a subset for faster comparison (comment out for full dataset)
# MAX_SAMPLES=10

mkdir -p ${OUTPUT_DIR}

echo "============================================================================"
echo "Comparing AP Evaluation Methods"
echo "============================================================================"
echo "Checkpoint: ${CHECKPOINT}"
echo "Batch size: ${BATCH_SIZE}"
echo ""

# Method 1: Class-Agnostic AP (current approach)
# echo "============================================================================"
# echo "Method 1: Class-Agnostic AP (Original)"
# echo "============================================================================"

# python3 evaluate_ovigod_ap.py \
#     --checkpoint ${CHECKPOINT} \
#     --backend vllm \
#     --batch_size ${BATCH_SIZE} \
#     --num_workers ${NUM_WORKERS} \
#     --output_file ${OUTPUT_DIR}/class_agnostic_ap.json

# echo ""
# echo "Method 1 completed!"
# echo ""

# Method 2: Per-Affordance mAP (like PF-Florence)
# echo "============================================================================"
# echo "Method 2: Per-Affordance mAP (PF-Florence style)"
# echo "============================================================================"

# python3 evaluate_ovigod_ap_per_affordance.py \
#     --checkpoint ${CHECKPOINT} \
#     --backend vllm \
#     --batch_size ${BATCH_SIZE} \
#     --num_workers ${NUM_WORKERS} \
#     --output_file ${OUTPUT_DIR}/rex_omni_per_affordance_map.json

# echo ""
# echo "Method 2 completed!"
# echo ""

python3 evaluate_ovigod_ap_per_affordance_ensemble.py \
    --checkpoint ${CHECKPOINT} \
    --backend vllm \
    --batch_size ${BATCH_SIZE} \
    --num_workers ${NUM_WORKERS} \
    --n_samples 4 \
    --vote_threshold 0.5 \
    --iou_threshold 0.5 \
    --temperature 1.0 \
    --output_file ${OUTPUT_DIR}/per_affordance_ensemble_map_n4_vote0.5_iou0.5_temp1.0.json
# Compare results
# echo "============================================================================"
# echo "Comparison Results"
# echo "============================================================================"

# python3 << 'EOF'
# import json
# import sys

# output_dir = "ap_method_comparison"

# try:
#     # Load results
#     with open(f"{output_dir}/class_agnostic_ap.json") as f:
#         method1 = json.load(f)
    
#     with open(f"{output_dir}/per_affordance_map.json") as f:
#         method2 = json.load(f)
    
#     print("\n" + "="*80)
#     print("COMPARISON: Class-Agnostic AP vs Per-Affordance mAP")
#     print("="*80)
    
#     print("\n📊 Method 1: Class-Agnostic AP (Original)")
#     print("   • All affordances in one sample")
#     print("   • Predictions can match any GT box (regardless of affordance)")
#     print("   • Evaluates: Pure localization ability")
#     print(f"   • AP@50:    {method1['metrics']['AP@50']:.4f}")
#     print(f"   • AP@75:    {method1['metrics']['AP@75']:.4f}")
#     print(f"   • AP@50:95: {method1['metrics']['AP@50:95']:.4f}")
    
#     print("\n📊 Method 2: Per-Affordance mAP (PF-Florence style)")
#     print("   • Each affordance as separate sample")
#     print("   • Predictions only match same-affordance GT boxes")
#     print("   • Evaluates: Localization + Classification ability")
#     print(f"   • mAP@50:    {method2['mAP_metrics']['mAP@50']:.4f}")
#     print(f"   • mAP@75:    {method2['mAP_metrics']['mAP@75']:.4f}")
#     print(f"   • mAP@50:95: {method2['mAP_metrics']['mAP@50:95']:.4f}")
    
#     print("\n📈 Difference (Method 1 - Method 2):")
#     diff_50 = method1['metrics']['AP@50'] - method2['mAP_metrics']['mAP@50']
#     diff_75 = method1['metrics']['AP@75'] - method2['mAP_metrics']['mAP@75']
#     diff_50_95 = method1['metrics']['AP@50:95'] - method2['mAP_metrics']['mAP@50:95']
    
#     print(f"   • Δ AP@50:    {diff_50:+.4f} ({diff_50/method2['mAP_metrics']['mAP@50']*100:+.2f}%)")
#     print(f"   • Δ AP@75:    {diff_75:+.4f} ({diff_75/method2['mAP_metrics']['mAP@75']*100:+.2f}%)")
#     print(f"   • Δ AP@50:95: {diff_50_95:+.4f} ({diff_50_95/method2['mAP_metrics']['mAP@50:95']*100:+.2f}%)")
    
#     # Per-affordance breakdown
#     print("\n📋 Per-Affordance AP Breakdown (Method 2):")
#     print(f"   {'Affordance':<20} {'AP@50':>10} {'AP@75':>10} {'AP@50:95':>10}")
#     print("   " + "-"*52)
    
#     per_aff = method2['per_affordance_metrics']
#     for affordance in sorted(per_aff.keys()):
#         stats = per_aff[affordance]
#         print(f"   {affordance:<20} {stats['AP@50']:>10.4f} {stats['AP@75']:>10.4f} {stats['AP@50:95']:>10.4f}")
    
#     print("\n💡 Interpretation:")
#     if diff_50_95 > 0.01:
#         print(f"   ✓ Method 1 is {abs(diff_50_95/method2['mAP_metrics']['mAP@50:95']*100):.1f}% higher")
#         print("   ✓ This suggests some predictions have correct location but wrong affordance")
#         print("   ✓ The model is better at localization than affordance classification")
#     elif diff_50_95 < -0.01:
#         print(f"   ⚠ Method 1 is {abs(diff_50_95/method2['mAP_metrics']['mAP@50:95']*100):.1f}% lower")
#         print("   ⚠ This is unexpected - method 1 should be equal or higher")
#     else:
#         print("   ≈ Both methods give similar results")
#         print("   ≈ The model rarely predicts correct location with wrong affordance")
    
#     print("\n" + "="*80)
    
# except FileNotFoundError as e:
#     print(f"\n❌ Error: Could not find result files")
#     print(f"   {e}")
#     sys.exit(1)
# except Exception as e:
#     print(f"\n❌ Error during comparison: {e}")
#     import traceback
#     traceback.print_exc()
#     sys.exit(1)

# EOF

# echo ""
# echo "All comparisons completed! Results saved in ${OUTPUT_DIR}/"

