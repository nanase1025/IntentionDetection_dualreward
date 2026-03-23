from dataset.task_fns import GroundingTaskFn
from dataset.task_fns.task_prompts.grounding_task import (
    GROUNDING_SINGLE_REGION_STAGE_XYXY,
)
from verl.utils.dataset import TSVRLHFDataset

min_pixels = 16 * 28 * 28
max_pixels = 2560 * 28 * 28

# ========================================
# GRPO VLM Semantic Reward Configuration (IoU < 0.6)
# ========================================
# This config uses samples with IoU < 0.6 from SFT evaluation
# Uses InternVL3.5 1B for captioning + BGE M3 for semantic similarity
#
# Pipeline:
#   1. Crop bbox region from image
#   2. InternVL3.5 1B generates caption
#   3. BGE M3 computes similarity(caption, intention_query)
#   4. Reward = 1 if similarity > 0.5, else 0
#
# NOTE: VLM Semantic Reward parameters configured in reward_func.py
# Default configuration:
#   - similarity_threshold: 0.5


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
    dataset_name="coco_outdoor_intention_grpo_vlm",
    reward_name="vlm_semantic",  # ✨ Use VLM Semantic Reward (InternVL + BGE)
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
    dataset_name="scannet_intention_grpo_vlm",
    reward_name="vlm_semantic",  # ✨ Use VLM Semantic Reward (InternVL + BGE)
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
    dataset_name="egoobject_intention_grpo_vlm",
    reward_name="vlm_semantic",  # ✨ Use VLM Semantic Reward (InternVL + BGE)
)

# Combine all three filtered datasets for GRPO training with VLM Semantic Reward
train_dataset = [
    coco_outdoor_data,  # 1,427 samples (IoU < 0.6)
    scannet_data,       # 3,011 samples (IoU < 0.6)
    egoobject_data,     # 2,186 samples (IoU < 0.6)
    # Total: 6,624 samples (24.61% of original 26,911)
    # These are the most difficult samples that benefit most from semantic guidance
]
