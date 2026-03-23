from dataset.task_fns import GroundingTaskFn
from dataset.task_fns.task_prompts.grounding_task import (
    GROUNDING_SINGLE_REGION_STAGE_XYXY,
)
from verl.utils.dataset import TSVRLHFDataset

min_pixels = 16 * 28 * 28
max_pixels = 2560 * 28 * 28

# ========================================
# GRPO Dual Reward Configuration (IoU < 0.8)
# ========================================
# This config uses samples with IoU < 0.8 from SFT evaluation
# Combined with CLIP semantic reward for better intention understanding

# NOTE: Dual Reward parameters are configured in reward_func.py
# Default configuration:
#   - alpha: 0.5 (IoU weight)
#   - beta: 0.5 (CLIP weight)
#   - iou_threshold: 0.5
#   - clip_threshold: 22.0


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
    dataset_name="coco_outdoor_intention_grpo_dual",
    reward_name="dual",  # ✨ Use Dual Reward (IoU + CLIP)
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
    dataset_name="scannet_intention_grpo_dual",
    reward_name="dual",  # ✨ Use Dual Reward (IoU + CLIP)
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
    dataset_name="egoobject_intention_grpo_dual",
    reward_name="dual",  # ✨ Use Dual Reward (IoU + CLIP)
)

# Combine all three filtered datasets for GRPO training with Dual Reward
train_dataset = [
    coco_outdoor_data,  # 1,427 samples (IoU < 0.6)
    scannet_data,       # 3,011 samples (IoU < 0.6)
    egoobject_data,     # 2,186 samples (IoU < 0.6)
    # Total: 6,624 samples (24.61% of original 26,911)
    # These are the most difficult samples that benefit most from dual reward guidance
]
