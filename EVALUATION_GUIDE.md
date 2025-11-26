# OV-IGOD 评估方法指南

本文档说明不同评估脚本的区别和使用方法。

## 📊 评估方法对比

### 1. Class-Agnostic AP (原始方法)
**脚本**: `evaluate_ovigod_ap.py`

**特点**:
- 所有 affordance 在一张图片中一起评估
- 预测框可以匹配任何 GT 框（不管 affordance 是否一致）
- 评估的是：**纯定位能力**
- 更容易获得高分（因为不需要分类）

**用法**:
```bash
python3 evaluate_ovigod_ap.py \
    --checkpoint finetuning/work_dirs/ovigod_sft_5ep \
    --backend vllm \
    --max_samples 100
```

---

### 2. Per-Affordance mAP (PF-Florence 风格)
**脚本**: `evaluate_ovigod_ap_per_affordance.py`

**特点**:
- 每个 affordance 作为单独的样本评估
- 预测框只能匹配相同 affordance 的 GT 框
- 评估的是：**定位 + 分类能力**
- 最终计算所有 affordance 的平均 AP (mAP)

**用法**:
```bash
python3 evaluate_ovigod_ap_per_affordance.py \
    --checkpoint finetuning/work_dirs/ovigod_sft_5ep \
    --backend vllm \
    --max_samples 100
```

**输出示例**:
```
mAP@50:     0.7500 (75.00%)
mAP@75:     0.6000 (60.00%)
mAP@50:95:  0.5500 (55.00%)

Per-Affordance AP Breakdown:
Affordance                                   AP@50      AP@75   AP@50:95
--------------------------------------------------------------------
I need a comfortable place to sit...        0.8500     0.7000     0.6500
I need a place to organize toys...          0.6500     0.5000     0.4500
...
```

---

### 3. Class-Agnostic AP with Ensemble
**脚本**: `evaluate_ovigod_ap_ensemble.py`

**特点**:
- Class-Agnostic 评估方式
- 使用 Temperature Ensemble + Voting 提升性能
- 多次推理，通过投票选择可靠的预测框
- 可以提供 confidence score（投票比例）

**用法**:
```bash
python3 evaluate_ovigod_ap_ensemble.py \
    --checkpoint finetuning/work_dirs/ovigod_sft_5ep \
    --backend vllm \
    --n_samples 5 \
    --vote_threshold 0.4 \
    --temperature 1.0 \
    --max_samples 100
```

**参数说明**:
- `n_samples`: 每张图片推理次数（ensemble 大小）
- `vote_threshold`: 投票阈值，例如 0.4 表示框必须出现在 40% 的预测中才保留
- `temperature`: 采样温度，1.0 匹配训练时的分布
- `iou_threshold`: 聚类相似框的 IoU 阈值

---

### 4. Per-Affordance mAP with Ensemble (最全面的方法)
**脚本**: `evaluate_ovigod_ap_per_affordance_ensemble.py`

**特点**:
- 结合 Per-Affordance 和 Ensemble 两种方法
- 每个 affordance 单独评估 + 多次推理投票
- 评估最全面：**定位 + 分类 + 不确定性估计**
- 最接近实际应用场景

**用法**:
```bash
python3 evaluate_ovigod_ap_per_affordance_ensemble.py \
    --checkpoint finetuning/work_dirs/ovigod_sft_5ep \
    --backend vllm \
    --n_samples 5 \
    --vote_threshold 0.4 \
    --temperature 1.0 \
    --max_samples 100
```

**快速测试**:
```bash
./run_per_affordance_ensemble.sh
```

---

## 🚀 快速测试脚本

### 比较 Class-Agnostic vs Per-Affordance
```bash
./compare_ap_methods.sh
```

### 测试不同的 Ensemble 配置
```bash
# Per-Affordance + Ensemble
./test_per_affordance_ensemble_configs.sh

# Class-Agnostic + Ensemble
./test_ensemble_configs.sh
```

---

## 📈 如何选择评估方法？

### 研究/论文
推荐使用：**Per-Affordance mAP** (`evaluate_ovigod_ap_per_affordance.py`)
- 更公平的比较（与 PF-Florence 等方法一致）
- 能看到每个 affordance 的性能
- mAP 是目标检测的标准指标

### 模型调优/超参数搜索
推荐使用：**Class-Agnostic AP** (`evaluate_ovigod_ap.py`)
- 运行速度快（不需要拆分样本）
- 快速评估定位能力
- 适合快速迭代

### GRPO 训练后评估
推荐使用：**Per-Affordance mAP with Ensemble** (`evaluate_ovigod_ap_per_affordance_ensemble.py`)
- 匹配训练时的采样策略（temperature > 0）
- 通过 ensemble 获得更稳定的结果
- 可以评估模型的不确定性
- 推荐参数：
  - `temperature=1.0` (匹配训练分布)
  - `n_samples=5` (合理的 ensemble 大小)
  - `vote_threshold=0.4` (40% 投票阈值)

### 最终性能报告
推荐同时报告：
1. **Per-Affordance mAP** (贪婪解码，temperature=0)
2. **Per-Affordance mAP with Ensemble** (温度采样)

---

## 🔧 性能优化建议

### 加速推理
```bash
# 增大 batch size
--batch_size 32

# 增加数据加载线程
--num_workers 16

# 使用 vLLM backend
--backend vllm
```

### 内存优化
```bash
# 减小 batch size
--batch_size 8

# 减少 ensemble 大小
--n_samples 3
```

### 快速测试
```bash
# 只测试少量样本
--max_samples 10
```

---

## 📊 结果解读

### AP 指标含义
- **AP@50**: IoU ≥ 0.5 时的平均精度
- **AP@75**: IoU ≥ 0.75 时的平均精度（更严格）
- **AP@50:95**: IoU 从 0.5 到 0.95（步长 0.05）的平均 AP

### Class-Agnostic vs Per-Affordance 差异
- 如果 Class-Agnostic AP 显著高于 Per-Affordance mAP：
  - 模型定位能力好，但分类能力弱
  - 很多框位置正确但 affordance 错误
  
- 如果两者接近：
  - 模型分类能力好
  - 很少预测错误的 affordance

### Ensemble 效果
- 如果 Ensemble 显著提升性能：
  - 模型输出不确定性较高
  - 温度采样 + 投票可以过滤噪声预测
  
- 如果 Ensemble 效果不明显：
  - 模型已经很稳定
  - 贪婪解码已经接近最优

---

## 💡 常见问题

### Q: 为什么 Per-Affordance 需要更长时间？
A: 因为每个 affordance 都作为单独的样本，样本数量会增加。例如一张图片有 5 个 affordance，会变成 5 个样本。

### Q: Ensemble 会增加多少推理时间？
A: 推理时间约等于 `n_samples` 倍。例如 `n_samples=5` 会增加 5 倍推理时间。但使用 vLLM 的 batch 推理可以减少总时间。

### Q: 应该使用多大的 n_samples？
A: 建议：
- 快速测试：`n_samples=3`
- 平衡性能和速度：`n_samples=5`
- 最佳性能：`n_samples=7~10`

### Q: vote_threshold 应该设置多少？
A: 建议：
- 宽松（召回率优先）：`0.3-0.4`
- 平衡：`0.4-0.5`
- 严格（精确率优先）：`0.5-0.6`

---

## 📝 完整示例

### 完整评估流程
```bash
# 1. 快速测试（10 张图片）
python3 evaluate_ovigod_ap_per_affordance.py \
    --checkpoint finetuning/work_dirs/ovigod_sft_5ep \
    --backend vllm \
    --max_samples 10

# 2. 测试 ensemble 配置（50 张图片）
./test_per_affordance_ensemble_configs.sh

# 3. 使用最佳配置评估全部数据集
python3 evaluate_ovigod_ap_per_affordance_ensemble.py \
    --checkpoint finetuning/work_dirs/ovigod_sft_5ep \
    --backend vllm \
    --n_samples 5 \
    --vote_threshold 0.4 \
    --temperature 1.0 \
    --batch_size 32 \
    --num_workers 16 \
    --output_file final_results.json

# 4. 也运行贪婪解码版本做对比
python3 evaluate_ovigod_ap_per_affordance.py \
    --checkpoint finetuning/work_dirs/ovigod_sft_5ep \
    --backend vllm \
    --batch_size 32 \
    --num_workers 16 \
    --output_file final_results_greedy.json
```

---

## 📚 参考

- PF-Florence 论文使用 Per-Affordance mAP
- COCO 评估使用 class-aware AP
- Ensemble 方法参考 Test-Time Augmentation (TTA)

