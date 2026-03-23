from dataset.task_fns import GroundingTaskFn
from dataset.task_fns.task_prompts.grounding_task import (
    GROUNDING_SINGLE_REGION_STAGE_XYXY,
)
from verl.utils.dataset import TSVRLHFDataset

min_pixels = 16 * 28 * 28
max_pixels = 2560 * 28 * 28

# ========================================
# GRPO Filtered Datasets Configuration (IoU < 0.6)
# ========================================
# This config uses ONLY samples with IoU < 0.6 from SFT evaluation
# These are the most difficult samples that need improvement through GRPO


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
    dataset_name="coco_outdoor_intention_grpo",
    reward_name="box_iou",  # Use IoU-based reward for bounding box detection
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
    dataset_name="scannet_intention_grpo",
    reward_name="box_iou",
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
    dataset_name="egoobject_intention_grpo",
    reward_name="box_iou",
)

# Combine all three filtered datasets for GRPO training
# Note: GRPO uses a list format, not ConcatDataset
train_dataset = [
    coco_outdoor_data,  # 1,427 samples (IoU < 0.6)
    scannet_data,       # 3,011 samples (IoU < 0.6)
    egoobject_data,     # 2,186 samples (IoU < 0.6)
    # Total: 6,624 samples (24.61% of original 26,911)
    # These are the most difficult samples where the SFT model performed poorly
    # and need improvement through GRPO
]
