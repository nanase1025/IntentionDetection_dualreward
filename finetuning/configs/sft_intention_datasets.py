from dataset import ConcatDataset, DataCollatorForSupervisedDataset, GroundingTSVDataset
from dataset.task_fns import GroundingTaskFn
from dataset.task_fns.task_prompts.grounding_task import (
    GROUNDING_SINGLE_REGION_STAGE_XYXY,
)

min_pixels = 16 * 28 * 28
max_pixels = 2560 * 28 * 28

model_name_or_path = "IDEA-Research/Rex-Omni"

# ========================================
# Three Intention Datasets Configuration
# ========================================

# 1. COCO Outdoor Intention Dataset
coco_outdoor_data = dict(
    type=GroundingTSVDataset,
    img_tsv_file="/home/hairong/hairong/data/intention_datasets_tsv_fixed/coco_outdoor_train.images.tsv",
    ann_tsv_file="/home/hairong/hairong/data/intention_datasets_tsv_fixed/coco_outdoor_train.annotations.tsv",
    ann_lineidx_file="/home/hairong/hairong/data/intention_datasets_tsv_fixed/coco_outdoor_train.annotations.tsv.lineidx",
    image_min_pixels=min_pixels,
    image_max_pixels=max_pixels,
    max_num_samples=None,  # Use all 8,000 samples
    task_fn=dict(
        type=GroundingTaskFn,
        task_prompts=GROUNDING_SINGLE_REGION_STAGE_XYXY,
        image_min_pixels=min_pixels,
        image_max_pixels=max_pixels,
    ),
    dataset_name="coco_outdoor_intention",
)

# 2. ScanNet Intention Dataset
scannet_data = dict(
    type=GroundingTSVDataset,
    img_tsv_file="/home/hairong/hairong/data/intention_datasets_tsv_fixed/scannet_train.images.tsv",
    ann_tsv_file="/home/hairong/hairong/data/intention_datasets_tsv_fixed/scannet_train.annotations.tsv",
    ann_lineidx_file="/home/hairong/hairong/data/intention_datasets_tsv_fixed/scannet_train.annotations.tsv.lineidx",
    image_min_pixels=min_pixels,
    image_max_pixels=max_pixels,
    max_num_samples=None,  # Use all 7,383 samples
    task_fn=dict(
        type=GroundingTaskFn,
        task_prompts=GROUNDING_SINGLE_REGION_STAGE_XYXY,
        image_min_pixels=min_pixels,
        image_max_pixels=max_pixels,
    ),
    dataset_name="scannet_intention",
)

# 3. EgoObject Intention Dataset
egoobject_data = dict(
    type=GroundingTSVDataset,
    img_tsv_file="/home/hairong/hairong/data/intention_datasets_tsv_fixed/egoobject_train.images.tsv",
    ann_tsv_file="/home/hairong/hairong/data/intention_datasets_tsv_fixed/egoobject_train.annotations.tsv",
    ann_lineidx_file="/home/hairong/hairong/data/intention_datasets_tsv_fixed/egoobject_train.annotations.tsv.lineidx",
    image_min_pixels=min_pixels,
    image_max_pixels=max_pixels,
    max_num_samples=None,  # Use all 11,528 samples
    task_fn=dict(
        type=GroundingTaskFn,
        task_prompts=GROUNDING_SINGLE_REGION_STAGE_XYXY,
        image_min_pixels=min_pixels,
        image_max_pixels=max_pixels,
    ),
    dataset_name="egoobject_intention",
)

# Combine all three datasets for training
train_dataset = dict(
    type=ConcatDataset,
    datasets=[
        coco_outdoor_data,  # 8,000 samples
        scannet_data,       # 7,383 samples
        egoobject_data,     # 11,528 samples
        # Total: 26,911 samples
    ],
)

data_collator = dict(type=DataCollatorForSupervisedDataset)

