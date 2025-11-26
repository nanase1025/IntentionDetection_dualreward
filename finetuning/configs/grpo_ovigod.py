from dataset.task_fns import GroundingTaskFn
from dataset.task_fns.task_prompts.grounding_task import (
    GROUNDING_SINGLE_REGION_STAGE_XYXY,
)
from verl.utils.dataset import TSVRLHFDataset

min_pixels = 16 * 28 * 28
max_pixels = 2560 * 28 * 28


# OV-IGOD dataset for GRPO training (affordance-based grounding)
ovigod_grpo_data = dict(
    type=TSVRLHFDataset,
    image_tsv_file="/workspace/hairong/data/ov-igod-dataset/train.images.tsv",
    anno_tsv_file="/workspace/hairong/data/ov-igod-dataset/train.annotations.tsv",
    anno_idx_file="/workspace/hairong/data/ov-igod-dataset/train.annotations.tsv.lineidx",
    min_pixels=min_pixels,
    max_pixels=max_pixels,
    task_fn=dict(
        type=GroundingTaskFn,
        task_prompts=GROUNDING_SINGLE_REGION_STAGE_XYXY,
        image_min_pixels=min_pixels,
        image_max_pixels=max_pixels,
    ),
    dataset_name="ovigod_affordance_grounding",
    reward_name="box_iou",  # Use IoU-based reward for bounding box detection
)

train_dataset = [
    ovigod_grpo_data,
]

