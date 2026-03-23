from dataset.task_fns import GroundingTaskFn
from dataset.task_fns.task_prompts.grounding_task import (
    GROUNDING_SINGLE_REGION_STAGE_XYXY,
)
from verl.utils.dataset import TSVRLHFDataset

min_pixels = 16 * 28 * 28
max_pixels = 2560 * 28 * 28

# ========================================
# GRPO IoU + VLM Hybrid Reward Configuration (IoU < 0.6)
# ========================================
# This config uses samples with IoU < 0.6 from SFT evaluation
# Combined with VLM semantic reward (InternVL + BGE M3) for comprehensive evaluation
#
# Hybrid Reward = alpha * IoU_reward + beta * VLM_reward
# Default: alpha=0.5, beta=0.5
#
# Pipeline:
#   1. IoU Reward: Measures bbox position accuracy (binary 0 or 1)
#   2. VLM Reward: InternVL generates caption → BGE M3 similarity (binary 0 or 1)
#   3. Final = 0.5 × IoU + 0.5 × VLM
#
# NOTE: IoU+VLM parameters configured in reward_func.py
# Default configuration:
#   - alpha: 0.5 (IoU weight)
#   - beta: 0.5 (VLM weight)
#   - iou_threshold: 0.5
#   - vlm_threshold: 0.5


# 1. COCO Outdoor Intention Dataset (IoU < 0.6)
coco_outdoor_data = dict(
    type=TSVRLHFDataset,
    image_tsv_file="/home/hairong/hairong/data/intention_datasets_tsv_fixed_grpo_iou_lt_06/coco_outdoor_train_grpo.images.tsv",
    anno_tsv_file="/home/hairong/hairong/data/intention_datasets_tsv_fixed_grpo_iou_lt_06/coco_outdoor_train_grpo.annotations.tsv",
    anno_idx_file="/home/hairong/hairong/data/intention_datasets_tsv_fixed_grpo_iou_lt_06/coco_outdoor_train_grpo.annotations.tsv.lineidx",
    min_pixels=min_pixels,
    max_pixels=max_pixels,
    task_fn=dict(
        type=GroundingTaskFn,
        task_prompts=GROUNDING_SINGLE_REGION_STAGE_XYXY,
        image_min_pixels=min_pixels,
        image_max_pixels=max_pixels,
    ),
    dataset_name="coco_outdoor_intention_grpo_iou_vlm",
    reward_name="iou_vlm",  # ✨ Use IoU + VLM Hybrid Reward
)


# 2. ScanNet Intention Dataset (IoU < 0.6)
scannet_data = dict(
    type=TSVRLHFDataset,
    image_tsv_file="/home/hairong/hairong/data/intention_datasets_tsv_fixed_grpo_iou_lt_06/scannet_train_grpo.images.tsv",
    anno_tsv_file="/home/hairong/hairong/data/intention_datasets_tsv_fixed_grpo_iou_lt_06/scannet_train_grpo.annotations.tsv",
    anno_idx_file="/home/hairong/hairong/data/intention_datasets_tsv_fixed_grpo_iou_lt_06/scannet_train_grpo.annotations.tsv.lineidx",
    min_pixels=min_pixels,
    max_pixels=max_pixels,
    task_fn=dict(
        type=GroundingTaskFn,
        task_prompts=GROUNDING_SINGLE_REGION_STAGE_XYXY,
        image_min_pixels=min_pixels,
        image_max_pixels=max_pixels,
    ),
    dataset_name="scannet_intention_grpo_iou_vlm",
    reward_name="iou_vlm",  # ✨ Use IoU + VLM Hybrid Reward
)


# 3. EgoObject Intention Dataset (IoU < 0.6)
egoobject_data = dict(
    type=TSVRLHFDataset,
    image_tsv_file="/home/hairong/hairong/data/intention_datasets_tsv_fixed_grpo_iou_lt_06/egoobject_train_grpo.images.tsv",
    anno_tsv_file="/home/hairong/hairong/data/intention_datasets_tsv_fixed_grpo_iou_lt_06/egoobject_train_grpo.annotations.tsv",
    anno_idx_file="/home/hairong/hairong/data/intention_datasets_tsv_fixed_grpo_iou_lt_06/egoobject_train_grpo.annotations.tsv.lineidx",
    min_pixels=min_pixels,
    max_pixels=max_pixels,
    task_fn=dict(
        type=GroundingTaskFn,
        task_prompts=GROUNDING_SINGLE_REGION_STAGE_XYXY,
        image_min_pixels=min_pixels,
        image_max_pixels=max_pixels,
    ),
    dataset_name="egoobject_intention_grpo_iou_vlm",
    reward_name="iou_vlm",  # ✨ Use IoU + VLM Hybrid Reward
)

# Combine all three filtered datasets for GRPO training with IoU + VLM Hybrid Reward
train_dataset = [
    coco_outdoor_data,  # 1,427 samples (IoU < 0.6)
    scannet_data,       # 3,011 samples (IoU < 0.6)
    egoobject_data,     # 2,186 samples (IoU < 0.6)
    # Total: 6,624 samples (24.61% of original 26,911)
    # These are the most difficult samples that benefit most from hybrid reward guidance
]
