from dataset import ConcatDataset, DataCollatorForSupervisedDataset, GroundingTSVDataset
from dataset.task_fns import GroundingTaskFn
from dataset.task_fns.task_prompts.grounding_task import (
    GROUNDING_SINGLE_REGION_STAGE_XYXY,
)

min_pixels = 16 * 28 * 28
max_pixels = 2560 * 28 * 28

model_name_or_path = "IDEA-Research/Rex-Omni"

# OV-IGOD 数据集配置 - 基于意图/功能描述的目标检测
ovigod_data = dict(
    type=GroundingTSVDataset,
    img_tsv_file="/home/hairong/hairong/data/ov-igod-dataset/train.images.tsv",
    ann_tsv_file="/home/hairong/hairong/data/ov-igod-dataset/train.annotations.tsv",
    ann_lineidx_file="/home/hairong/hairong/data/ov-igod-dataset/train.annotations.tsv.lineidx",
    image_min_pixels=min_pixels,
    image_max_pixels=max_pixels,
    max_num_samples=None,  # 快速测试：只使用1000个样本（设为None则使用全部6701个）
    task_fn=dict(
        type=GroundingTaskFn,
        task_prompts=GROUNDING_SINGLE_REGION_STAGE_XYXY,
        image_min_pixels=min_pixels,
        image_max_pixels=max_pixels,
    ),
    dataset_name="ovigod_affordance_grounding",
)

train_dataset = dict(
    type=ConcatDataset,
    datasets=[
        ovigod_data,
    ],
)

data_collator = dict(type=DataCollatorForSupervisedDataset)

