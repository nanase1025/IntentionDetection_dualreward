# Temperature Ensemble Evaluation Guide

## 🎯 核心思想

**问题**：Rex-Omni 是自回归模型，不输出置信度分数，所有预测框被平等对待。

**解决方案**：Temperature Ensemble + Voting
1. 用 `temperature=1.0`（和 GRPO 训练时一样）生成 n 次预测
2. 使用 IoU 聚类找到"相似"的框
3. 出现频率作为"伪置信度" → `confidence = vote_count / n_samples`
4. 只保留高频框（`vote_ratio > threshold`）

## 📁 文件说明

| 文件 | 功能 |
|------|------|
| `evaluate_ovigod_ap_ensemble.py` | 主评估脚本（带 ensemble） |
| `test_ensemble_configs.sh` | 快速测试多种配置 |
| `visualize_ensemble_sample.py` | 可视化单个样本的 ensemble 过程 |

## 🚀 快速开始

### 1. 单次评估（自定义配置）

```bash
python evaluate_ovigod_ap_ensemble.py \
    --checkpoint finetuning/work_dirs/ovigod_sft/checkpoint-627 \
    --max_samples 100 \
    --n_samples 5 \
    --vote_threshold 0.4 \
    --iou_threshold 0.5 \
    --temperature 1.0 \
    --backend vllm \
    --output_file ensemble_n5_vote0.4.json
```

**参数说明**：
- `--n_samples`: 每张图生成几次预测（5-10 推荐）
- `--vote_threshold`: 最小投票率，保留出现 ≥ 40% 的框
- `--iou_threshold`: IoU 阈值，判断两个框是否"相同"
- `--temperature`: 采样温度（1.0 = 训练时分布）

### 2. 批量测试多种配置

```bash
./test_ensemble_configs.sh
```

会自动测试：
- Baseline: `temp=0, n=1`（当前评估方式）
- Config 1: `temp=1.0, n=3, vote=0.4`
- Config 2: `temp=1.0, n=5, vote=0.4`
- Config 3: `temp=1.0, n=5, vote=0.3`
- Config 4: `temp=0.5, n=5, vote=0.4`

结果保存在 `ensemble_experiments/`，脚本会自动对比并显示最佳配置。

### 3. 可视化单个样本

```bash
python visualize_ensemble_sample.py \
    --checkpoint finetuning/work_dirs/ovigod_sft/checkpoint-627 \
    --image_path data/ov-igod-dataset/sunrgbd_jpgs/1.jpg \
    --category "I want to sit" \
    --n_samples 10 \
    --vote_threshold 0.4 \
    --output ensemble_viz.png
```

生成三栏可视化：
1. **所有原始预测**（不同采样用不同颜色）
2. **聚类结果**（每个 cluster 一种颜色 + 投票数）
3. **最终预测**（只保留投票 > 阈值的框）

## 📊 预期效果

### 优点 ✅

1. **获得置信度**
   ```python
   # Before (无置信度)
   predictions = [box1, box2, box3]  # 无法判断哪个更可靠
   
   # After (有置信度)
   predictions = [
       {'coords': box1, 'confidence': 0.8},  # 8/10 次都预测到
       {'coords': box2, 'confidence': 0.5},  # 5/10 次
       {'coords': box3, 'confidence': 0.3},  # 3/10 次 → 被过滤
   ]
   ```

2. **更鲁棒**
   - 过滤掉不稳定的预测（hallucination）
   - 一致性高的框被保留

3. **符合训练分布**
   - 测试时 temp=1.0 和训练时一致
   - 减少 train-test 分布差异

### 缺点 ❌

1. **速度慢**
   - `n=5` → 5倍推理时间
   - 2630张图 × 5次 ≈ 需要更长时间

2. **可能漏检**
   - 正确但只出现1-2次的框会被过滤
   - 需要调整 `vote_threshold`

## 🧪 实验建议

### 阶段 1：快速验证（100张图）

```bash
# 先测试是否有提升
./test_ensemble_configs.sh  # 约30-60分钟
```

**预期**：
- 如果 AP@50 提升 > 2%，继续
- 如果下降或持平，停止

### 阶段 2：参数调优

如果阶段1有效，测试更多配置：

```bash
# Test different vote thresholds
for vote in 0.2 0.3 0.4 0.5; do
    python evaluate_ovigod_ap_ensemble.py \
        --n_samples 5 \
        --vote_threshold $vote \
        ...
done

# Test different n_samples
for n in 3 5 7 10; do
    python evaluate_ovigod_ap_ensemble.py \
        --n_samples $n \
        --vote_threshold 0.4 \
        ...
done
```

### 阶段 3：全集评估

找到最优配置后，在全测试集（2630张）上评估：

```bash
python evaluate_ovigod_ap_ensemble.py \
    --checkpoint YOUR_BEST_MODEL \
    --max_samples null \  # All samples
    --n_samples 5 \
    --vote_threshold 0.4 \
    --output_file final_ensemble_results.json
```

## 📈 预期提升

基于类似方法的文献，可能的提升：

| 场景 | 预期 AP 提升 |
|------|-------------|
| **保守估计** | +1-2% |
| **乐观估计** | +3-5% |
| **理想情况** | +5-10% |

**取决于**：
- 模型的不确定性（temperature=1.0 时的多样性）
- 数据集难度（困难样本 ensemble 效果更好）
- 参数调优（找到最佳 n 和 threshold）

## 🔍 Debug 技巧

### 1. 检查多样性

```bash
# 可视化看看 n 次采样是否真的不同
python visualize_ensemble_sample.py --n_samples 10 ...
```

如果 10 次生成的框几乎完全一样 → ensemble 无效

### 2. 分析被过滤的框

修改 `vote_threshold`，看看哪些框被过滤：

```python
# 保存所有 cluster 信息
for cluster in clusters:
    vote_ratio = len(cluster) / n_samples
    print(f"Cluster: vote_ratio={vote_ratio}, boxes={len(cluster)}")
    
    if vote_ratio < vote_threshold:
        print(f"  → FILTERED (but maybe correct?)")
```

### 3. 速度优化

如果太慢，可以：
- 降低 `n_samples`（5 → 3）
- 只在困难样本上用 ensemble
- 使用 vLLM backend（已经最快）

## 💡 进阶想法

### 1. 加权投票

当前：所有样本权重相同

改进：根据 IoU 距离加权

```python
weight = sum(compute_iou(box, center) for box in cluster)
confidence = weight / n_samples  # 加权置信度
```

### 2. 自适应 n_samples

简单图像：n=3
复杂图像：n=10

根据图像特征（目标数量、遮挡程度）动态调整

### 3. 校准置信度

训练一个小的 MLP：`[vote_ratio, iou_std, box_size] → true_confidence`

使用验证集标注来学习更准确的置信度

## 📝 总结

**核心创新**：
- 从 temperature ensemble 中获得"伪置信度"
- 符合 GRPO 训练分布（temp=1.0）
- 简单有效，无需修改模型

**适用场景**：
- ✅ 自回归目标检测模型（无置信度输出）
- ✅ 需要更鲁棒的预测
- ✅ 可以接受慢一些的推理速度

**开始实验**：
```bash
./test_ensemble_configs.sh
```

祝实验顺利！🚀

