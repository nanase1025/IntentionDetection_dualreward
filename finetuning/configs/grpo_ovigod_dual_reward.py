from dataset.task_fns import GroundingTaskFn
from dataset.task_fns.task_prompts.grounding_task import (
    GROUNDING_SINGLE_REGION_STAGE_XYXY,
)
from verl.utils.dataset import TSVRLHFDataset

min_pixels = 16 * 28 * 28
max_pixels = 2560 * 28 * 28

# ========================================
# OV-IGOD GRPO Dual Reward Configuration
# ========================================
# Dual Reward = IoU F1 (position) + FG-CLIP (semantic)
# Full training set, no data filtering
#
# Dataset: OV-IGOD affordance-based grounding
#   - 6701 images, 15430 (image, affordance) samples
#
# Dual Reward parameters configured via environment variables:
#   - DUAL_REWARD_ALPHA: IoU weight (default 0.5)
#   - DUAL_REWARD_BETA: FG-CLIP weight (default 0.5)
#   - DUAL_REWARD_IOU_THRESHOLD: IoU threshold (default 0.5)
#   - DUAL_REWARD_CLIP_THRESHOLD: FG-CLIP threshold (default 15.0)
#   - CLIP_MODEL_NAME: FG-CLIP model (qihoo360/fg-clip-large)
#   - FGCLIP_IMAGE_SIZE: 336
#   - FGCLIP_USE_LONG_TEXT: true

ovigod_grpo_data = dict(
    type=TSVRLHFDataset,
    image_tsv_file="/home/hairong/hairong/data/ov-igod-dataset/train.images.tsv",
    anno_tsv_file="/home/hairong/hairong/data/ov-igod-dataset/train.annotations.tsv",
    anno_idx_file="/home/hairong/hairong/data/ov-igod-dataset/train.annotations.tsv.lineidx",
    min_pixels=min_pixels,
    max_pixels=max_pixels,
    task_fn=dict(
        type=GroundingTaskFn,
        task_prompts=GROUNDING_SINGLE_REGION_STAGE_XYXY,
        image_min_pixels=min_pixels,
        image_max_pixels=max_pixels,
    ),
    dataset_name="ovigod_affordance_grounding_dual",
    reward_name="dual",
)

train_dataset = [
    ovigod_grpo_data,
]
