from dataset.task_fns import GroundingTaskFn
from dataset.task_fns.task_prompts.grounding_task import (
    GROUNDING_SINGLE_REGION_STAGE_XYXY,
)
from verl.utils.dataset import TSVRLHFDataset

min_pixels = 16 * 28 * 28
max_pixels = 2560 * 28 * 28


# COCO Outdoor Intention dataset for GRPO training (FILTERED)
# Using samples with IoU in [0.2, 0.8] - medium difficulty
coco_outdoor_grpo_data = dict(
    type=TSVRLHFDataset,
    image_tsv_file="/home/hairong/hairong/code/IntentionDetection/finetuning/work_dirs/filtered_data/coco_outdoor_train_filtered.images.tsv",
    anno_tsv_file="/home/hairong/hairong/code/IntentionDetection/finetuning/work_dirs/filtered_data/coco_outdoor_train_filtered.annotations.tsv",
    anno_idx_file="/home/hairong/hairong/code/IntentionDetection/finetuning/work_dirs/filtered_data/coco_outdoor_train_filtered.annotations.tsv.lineidx",
    min_pixels=min_pixels,
    max_pixels=max_pixels,
    task_fn=dict(
        type=GroundingTaskFn,
        task_prompts=GROUNDING_SINGLE_REGION_STAGE_XYXY,
        image_min_pixels=min_pixels,
        image_max_pixels=max_pixels,
    ),
    dataset_name="coco_outdoor_intention_filtered",
    reward_name="box_iou",  # Use IoU-based reward for bounding box detection
)

# ScanNet Intention dataset for GRPO training (FILTERED)
# Using samples with IoU in [0.2, 0.8] - medium difficulty
scannet_grpo_data = dict(
    type=TSVRLHFDataset,
    image_tsv_file="/home/hairong/hairong/code/IntentionDetection/finetuning/work_dirs/filtered_data/scannet_train_filtered.images.tsv",
    anno_tsv_file="/home/hairong/hairong/code/IntentionDetection/finetuning/work_dirs/filtered_data/scannet_train_filtered.annotations.tsv",
    anno_idx_file="/home/hairong/hairong/code/IntentionDetection/finetuning/work_dirs/filtered_data/scannet_train_filtered.annotations.tsv.lineidx",
    min_pixels=min_pixels,
    max_pixels=max_pixels,
    task_fn=dict(
        type=GroundingTaskFn,
        task_prompts=GROUNDING_SINGLE_REGION_STAGE_XYXY,
        image_min_pixels=min_pixels,
        image_max_pixels=max_pixels,
    ),
    dataset_name="scannet_intention_filtered",
    reward_name="box_iou",
)

# EgoObject Intention dataset for GRPO training (FILTERED)
# Using samples with IoU in [0.2, 0.8] - medium difficulty
egoobject_grpo_data = dict(
    type=TSVRLHFDataset,
    image_tsv_file="/home/hairong/hairong/code/IntentionDetection/finetuning/work_dirs/filtered_data/egoobject_train_filtered.images.tsv",
    anno_tsv_file="/home/hairong/hairong/code/IntentionDetection/finetuning/work_dirs/filtered_data/egoobject_train_filtered.annotations.tsv",
    anno_idx_file="/home/hairong/hairong/code/IntentionDetection/finetuning/work_dirs/filtered_data/egoobject_train_filtered.annotations.tsv.lineidx",
    min_pixels=min_pixels,
    max_pixels=max_pixels,
    task_fn=dict(
        type=GroundingTaskFn,
        task_prompts=GROUNDING_SINGLE_REGION_STAGE_XYXY,
        image_min_pixels=min_pixels,
        image_max_pixels=max_pixels,
    ),
    dataset_name="egoobject_intention_filtered",
    reward_name="box_iou",
)

# Combine all three FILTERED datasets for GRPO training
# Total: 4,450 samples (COCO: 1,516, ScanNet: 1,834, EgoObject: 1,100)
# IoU range: [0.2, 0.8] - focusing on medium difficulty samples
train_dataset = [
    coco_outdoor_grpo_data,
    scannet_grpo_data,
    egoobject_grpo_data,
]
