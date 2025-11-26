# GRPO Training Guide for OV-IGOD Dataset

## Overview

GRPO (Generalized Reward Preference Optimization) is a reinforcement learning method that fine-tunes the model using reward signals. For object detection, it optimizes the model to maximize IoU scores between predicted and ground truth bounding boxes.

## Training Pipeline

```
SFT (Supervised Fine-Tuning)  →  GRPO (RL Optimization)
        ↓                                ↓
  AP@50: 42.30%                    Better localization
  AP@75: 30.11%                    Higher precision
  AP@50:95: 28.56%                 Improved recall
```

## Prerequisites

1. ✅ Completed SFT training
2. ✅ SFT checkpoint available at `work_dirs/ovigod_sft/checkpoint-627`
3. ✅ TSV dataset files ready
4. ✅ All required dependencies installed (including `verl`)

## How GRPO Works

### 1. Rollout Phase
- Model generates predictions for input images
- Multiple predictions (n=4) are sampled per prompt
- Predictions are in Rex-Omni's detection format

### 2. Reward Computation
- **Reward Function**: `box_iou` (F1-score based on IoU)
- Compares predicted boxes with ground truth
- Computes:
  - **Precision**: How many predicted boxes are correct
  - **Recall**: How many GT boxes are detected
  - **F1 Score**: Harmonic mean (used as reward)

### 3. Policy Update
- Uses GRPO algorithm to update model parameters
- Encourages actions (predictions) with high rewards
- Penalizes low-quality predictions
- KL divergence prevents drastic changes from SFT

## Configuration Files

### 1. Dataset Config: `configs/grpo_ovigod.py`
```python
# Defines the TSV dataset with reward function
ovigod_grpo_data = dict(
    type=TSVRLHFDataset,
    image_tsv_file="...",
    reward_name="box_iou",  # IoU-based reward
)
```

### 2. Training Config: `configs/grpo_ovigod.yaml`
Key parameters:
- **global_batch_size**: 16 (reduced for single GPU)
- **rollout.n**: 4 (number of samples per prompt)
- **learning_rate**: 5e-7 (much lower than SFT)
- **kl_coef**: 5e-3 (KL divergence penalty)

### 3. Launch Script: `scripts/grpo_ovigod.sh`
- Sets environment variables
- Configures single GPU training
- Enables debug logging

## Launch GRPO Training

### Step 1: Prepare Environment

```bash
cd /workspace/hairong/code/Rex-Omni/finetuning

# Make script executable
chmod +x scripts/grpo_ovigod.sh

# Verify SFT checkpoint exists
ls -lh work_dirs/ovigod_sft/checkpoint-627/
```

### Step 2: Run Training

```bash
# Option 1: Direct execution
cd /workspace/hairong/code/Rex-Omni/finetuning
./scripts/grpo_ovigod.sh

# Option 2: With nohup for background training
nohup ./scripts/grpo_ovigod.sh > grpo_training.log 2>&1 &

# Monitor progress
tail -f grpo_training.log
```

### Step 3: Monitor Training

**Wandb Dashboard**:
- Project: `rex-omni-grpo-ovigod`
- Metrics to watch:
  - `reward/mean`: Average reward (F1 score)
  - `reward/max`: Best reward in batch
  - `loss/policy_loss`: Policy gradient loss
  - `loss/kl`: KL divergence from SFT model

**Debug Logs**:
```bash
# View reward details
tail -f work_dirs/ovigod_grpo/log.txt

# Check visualizations (if enabled)
ls work_dirs/ovigod_grpo/visualizations/
```

## Understanding Rewards

The `box_iou` reward function computes:

1. **Parse Predictions**: Extract bounding boxes from model output
2. **Match Boxes**: For each GT box, find best matching prediction
3. **Calculate IoU**: Intersection over Union for matched pairs
4. **Compute F1**: 
   - Recall = Σ(best_iou_for_each_gt) / num_gt
   - Precision = Σ(best_iou_for_each_pred) / num_pred
   - **Reward = 2 * (Precision * Recall) / (Precision + Recall)**

### Example Reward Calculation

```
GT: 4 affordance boxes
Predictions: 5 detected boxes

Matched with IoU > 0.5: 3 boxes
- Recall = 3/4 = 0.75 (found 3 out of 4 GT)
- Precision = 3/5 = 0.60 (3 out of 5 predictions correct)
- F1 = 2 * (0.75 * 0.60) / (0.75 + 0.60) = 0.67

→ Reward = 0.67
```

## Training Parameters Explained

### Rollout Settings
```yaml
worker.rollout:
  n: 4  # Sample 4 predictions per prompt
  temperature: 1.0  # Sampling diversity
  top_p: 0.95  # Nucleus sampling
```
- Higher `n` → More exploration, slower training
- Lower `temperature` → More deterministic predictions

### Learning Rate
```yaml
worker.actor.optim:
  lr: 5.0e-7  # Much lower than SFT (2e-5)
```
- RL training requires smaller steps to stay stable
- Too high → Catastrophic forgetting of SFT
- Too low → Slow improvement

### KL Coefficient
```yaml
algorithm:
  kl_coef: 5.0e-3  # Penalty for diverging from SFT
```
- Prevents model from deviating too much from SFT
- Higher → Stays closer to SFT, conservative
- Lower → More aggressive optimization

## Checkpoints

GRPO saves checkpoints every 50 steps in `work_dirs/ovigod_grpo/`:

```
work_dirs/ovigod_grpo/
├── checkpoint-50/      # Early GRPO checkpoint
├── checkpoint-100/     # Middle checkpoint
├── checkpoint-150/     # Later checkpoint
├── log.txt            # Detailed reward logs
└── visualizations/    # Optional: GT vs Pred images
```

Each checkpoint contains:
- `pytorch_model.bin` or `model-*.safetensors`
- `config.json`
- `tokenizer files`
- `preprocessor_config.json` (should be copied from SFT)

## Evaluation

After GRPO training, evaluate on test set:

```bash
cd /workspace/hairong/code/Rex-Omni

# Test GRPO checkpoint
python evaluate_ovigod_ap.py \
    --checkpoint finetuning/work_dirs/ovigod_grpo/checkpoint-150 \
    --backend vllm \
    --output_file eval_ap_grpo.json

# Compare with SFT baseline
python evaluate_ovigod_ap.py \
    --checkpoint finetuning/work_dirs/ovigod_sft/checkpoint-627 \
    --backend vllm \
    --output_file eval_ap_sft.json
```

### Expected Improvements

GRPO typically improves:
- **Localization accuracy**: Higher AP@75 (stricter IoU)
- **Precision**: Fewer false positives
- **Bounding box quality**: Tighter boxes around objects

You might see:
```
SFT Results:
  AP@50: 42.30%
  AP@75: 30.11%  ← Should improve most
  AP@50:95: 28.56%

GRPO Results (expected):
  AP@50: 43-45%
  AP@75: 33-37%  ← Biggest gain
  AP@50:95: 30-33%
```

## Troubleshooting

### Issue 1: OOM (Out of Memory)

**Solution**: Reduce batch sizes in `configs/grpo_ovigod.yaml`
```yaml
worker.actor:
  global_batch_size: 8  # Reduce from 16
  micro_batch_size_per_device_for_update: 1  # Reduce from 2
worker.rollout:
  n: 2  # Reduce from 4
```

### Issue 2: Reward Always 0

**Causes**:
- Model not generating valid detection format
- Parsing error in reward function

**Debug**:
```bash
# Check log.txt for parsing errors
grep "Failed to parse" work_dirs/ovigod_grpo/log.txt

# Enable visualizations
export LOG_VISUALIZE_PATH="work_dirs/ovigod_grpo/vis"
```

### Issue 3: Training Unstable / Loss Spikes

**Solution**: Increase KL coefficient
```yaml
algorithm:
  kl_coef: 1.0e-2  # Increase from 5e-3
```

### Issue 4: Ray/VLLM Errors

**Solution**: Clear Ray cache and restart
```bash
# Kill all Ray processes
pkill -9 ray

# Clear temp files
rm -rf /tmp/ray/*

# Restart training
./scripts/grpo_ovigod.sh
```

### Issue 5: Missing preprocessor_config.json

**Solution**: Copy from SFT checkpoint
```bash
cp work_dirs/ovigod_sft/preprocessor_config.json \
   work_dirs/ovigod_grpo/checkpoint-50/
```

## Advanced: Hyperparameter Tuning

For better results, you can tune:

### 1. Reward Shaping
Modify `verl/configs/reward_func.py` to add bonus rewards:
```python
# Example: Bonus for correct category
if pred_class == gt_class:
    reward = iou * 1.2  # 20% bonus
```

### 2. Multi-step GRPO
Train in stages with decreasing KL:
```bash
# Stage 1: Conservative (high KL)
kl_coef: 1.0e-2
total_epochs: 1

# Stage 2: Aggressive (low KL)  
kl_coef: 2.0e-3
total_epochs: 1
```

### 3. Curriculum Learning
Start with easy samples, gradually add hard ones:
```python
# Filter dataset by image complexity
max_objects_per_image: 5  # Start
max_objects_per_image: 10  # Later
```

## Summary Commands

```bash
# 1. Start GRPO training
cd /workspace/hairong/code/Rex-Omni/finetuning
chmod +x scripts/grpo_ovigod.sh
CUDA_VISIBLE_DEVICES=7 ./scripts/grpo_ovigod.sh

# 2. Monitor progress
tail -f work_dirs/ovigod_grpo/log.txt
# or
wandb login
# then check https://wandb.ai

# 3. Evaluate best checkpoint
cd /workspace/hairong/code/Rex-Omni
python evaluate_ovigod_ap.py \
    --checkpoint finetuning/work_dirs/ovigod_grpo/checkpoint-150 \
    --backend vllm \
    --output_file eval_ap_grpo.json
```

## Key Differences: SFT vs GRPO

| Aspect | SFT | GRPO |
|--------|-----|------|
| **Learning** | Supervised (GT labels) | RL (Reward signal) |
| **Objective** | Minimize cross-entropy | Maximize reward |
| **Data** | Static GT annotations | Dynamic model rollouts |
| **Learning Rate** | 2e-5 | 5e-7 (100x smaller) |
| **Stability** | Stable | Can be unstable |
| **Improves** | General accuracy | Fine-grained localization |
| **When to use** | Always (first stage) | Optional (second stage) |

## References

- Rex-Omni Paper: [Link to paper]
- GRPO/REINFORCE Algorithm
- VERL Framework: https://github.com/volcengine/verl

Good luck with your GRPO training! 🚀

