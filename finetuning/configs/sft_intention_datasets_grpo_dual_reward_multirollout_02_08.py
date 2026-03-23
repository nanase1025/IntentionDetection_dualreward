from dataset.task_fns import GroundingTaskFn
from dataset.task_fns.task_prompts.grounding_task import (
    GROUNDING_SINGLE_REGION_STAGE_XYXY,
)
from verl.utils.dataset import TSVRLHFDataset

min_pixels = 16 * 28 * 28
max_pixels = 2560 * 28 * 28

# ========================================
# GRPO Dual Reward Configuration - Multi-Rollout Filtered (0.2 <= mean_iou < 0.8)
# ========================================
# This config uses samples filtered based on multi-rollout evaluation:
# - Excluded: Completely failed samples (mean_iou < 0.2)
# - Excluded: Already excellent samples (mean_iou >= 0.8)
# - Included: Samples with improvement potential (0.2 <= mean_iou < 0.8)
# - Combined with CLIP semantic reward for better intention understanding

# NOTE: Dual Reward parameters are configured via environment variables:
#   - DUAL_REWARD_ALPHA: IoU weight (default 0.5)
#   - DUAL_REWARD_BETA: CLIP weight (default 0.5)
#   - DUAL_REWARD_IOU_THRESHOLD: IoU threshold (default 0.5)
#   - DUAL_REWARD_CLIP_THRESHOLD: CLIP score threshold (default 22.0)


# 1. COCO Outdoor Intention Dataset (0.2 <= mean_iou < 0.8)
coco_outdoor_data = dict(
    type=TSVRLHFDataset,
    image_tsv_file="/home/hairong/hairong/data/intention_datasets_tsv_fixed_grpo_multirollout_02_08/coco_outdoor_train_grpo.images.tsv",
    anno_tsv_file="/home/hairong/hairong/data/intention_datasets_tsv_fixed_grpo_multirollout_02_08/coco_outdoor_train_grpo.annotations.tsv",
    anno_idx_file="/home/hairong/hairong/data/intention_datasets_tsv_fixed_grpo_multirollout_02_08/coco_outdoor_train_grpo.annotations.tsv.lineidx",
    min_pixels=min_pixels,
    max_pixels=max_pixels,
    task_fn=dict(
        type=GroundingTaskFn,
        task_prompts=GROUNDING_SINGLE_REGION_STAGE_XYXY,
        image_min_pixels=min_pixels,
        image_max_pixels=max_pixels,
    ),
    dataset_name="coco_outdoor_intention_grpo_dual_multirollout",
    reward_name="dual",  # ✨ Use Dual Reward (IoU + CLIP)
)


# 2. ScanNet Intention Dataset (0.2 <= mean_iou < 0.8)
scannet_data = dict(
    type=TSVRLHFDataset,
    image_tsv_file="/home/hairong/hairong/data/intention_datasets_tsv_fixed_grpo_multirollout_02_08/scannet_train_grpo.images.tsv",
    anno_tsv_file="/home/hairong/hairong/data/intention_datasets_tsv_fixed_grpo_multirollout_02_08/scannet_train_grpo.annotations.tsv",
    anno_idx_file="/home/hairong/hairong/data/intention_datasets_tsv_fixed_grpo_multirollout_02_08/scannet_train_grpo.annotations.tsv.lineidx",
    min_pixels=min_pixels,
    max_pixels=max_pixels,
    task_fn=dict(
        type=GroundingTaskFn,
        task_prompts=GROUNDING_SINGLE_REGION_STAGE_XYXY,
        image_min_pixels=min_pixels,
        image_max_pixels=max_pixels,
    ),
    dataset_name="scannet_intention_grpo_dual_multirollout",
    reward_name="dual",  # ✨ Use Dual Reward (IoU + CLIP)
)


# 3. EgoObject Intention Dataset (0.2 <= mean_iou < 0.8)
egoobject_data = dict(
    type=TSVRLHFDataset,
    image_tsv_file="/home/hairong/hairong/data/intention_datasets_tsv_fixed_grpo_multirollout_02_08/egoobject_train_grpo.images.tsv",
    anno_tsv_file="/home/hairong/hairong/data/intention_datasets_tsv_fixed_grpo_multirollout_02_08/egoobject_train_grpo.annotations.tsv",
    anno_idx_file="/home/hairong/hairong/data/intention_datasets_tsv_fixed_grpo_multirollout_02_08/egoobject_train_grpo.annotations.tsv.lineidx",
    min_pixels=min_pixels,
    max_pixels=max_pixels,
    task_fn=dict(
        type=GroundingTaskFn,
        task_prompts=GROUNDING_SINGLE_REGION_STAGE_XYXY,
        image_min_pixels=min_pixels,
        image_max_pixels=max_pixels,
    ),
    dataset_name="egoobject_intention_grpo_dual_multirollout",
    reward_name="dual",  # ✨ Use Dual Reward (IoU + CLIP)
)

# Combine all three filtered datasets for GRPO training with Dual Reward
train_dataset = [
    coco_outdoor_data,  # 1,473 samples (18.41% of 8,000)
    scannet_data,       # 1,919 samples (25.99% of 7,383)
    egoobject_data,     # 1,112 samples (9.65% of 11,528)
    # Total: 4,504 samples (16.74% of original 26,911)
    # 
    # These samples have:
    # - Some prediction capability (not completely failed)
    # - Room for improvement (not already excellent)
    # - Dual reward will improve both location AND semantic understanding
]
