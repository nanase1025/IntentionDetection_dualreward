# OV-IGOD 数据集训练指南

本指南介绍如何使用 OV-IGOD 数据集（基于意图/功能描述的目标检测）训练 Rex-Omni 模型。

## 📊 数据准备状态

✅ **已完成的步骤：**
1. 数据格式转换（6,701 个训练样本）
2. TSV 格式文件生成
3. 可视化验证（20个样本）
4. 训练配置文件创建
5. 训练脚本准备

## 📁 生成的文件

### 数据文件
- `/workspace/hairong/data/ov-igod-dataset/train.images.tsv` - 图片TSV文件
- `/workspace/hairong/data/ov-igod-dataset/train.annotations.tsv` - 标注TSV文件
- `/workspace/hairong/data/ov-igod-dataset/train.annotations.tsv.lineidx` - 索引文件
- `/workspace/hairong/data/ov-igod-dataset/vis/` - 可视化验证结果（20个样本）

### 训练文件
- `finetuning/configs/sft_ovigod.py` - 训练配置文件
- `finetuning/scripts/sft_ovigod.sh` - 训练启动脚本
- `test_ovigod_model.py` - 模型测试脚本

## 🚀 开始训练

### 1. 检查可视化结果（强烈推荐）

```bash
# 查看可视化结果，确认数据转换正确
ls /workspace/hairong/data/ov-igod-dataset/vis/
```

使用图片查看器打开 `vis/` 目录下的 PNG 文件，验证：
- ✅ 边界框位置是否正确
- ✅ 标签文本（意图描述）是否正确显示

### 2. 修改训练参数（根据实际GPU情况）

编辑 `finetuning/scripts/sft_ovigod.sh`：

```bash
# 修改GPU数量
GPUS_PER_NODE=8  # 改为你实际的GPU数量

# 如果显存不足，可以调整：
--per_device_train_batch_size 1  # 减小batch size
--gradient_accumulation_steps 16  # 增大梯度累积
```

### 3. 启动训练

```bash
cd /workspace/hairong/code/Rex-Omni/finetuning

# 使用8卡训练（默认配置）
bash scripts/sft_ovigod.sh

# 或者指定GPU
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/sft_ovigod.sh
```

### 4. 监控训练进度

```bash
# 方法1：查看日志
tail -f work_dirs/ovigod_sft/logs/*.log

# 方法2：使用 wandb（如果配置了）
# 访问 https://wandb.ai 查看训练曲线

# 方法3：检查checkpoint
ls work_dirs/ovigod_sft/
```

## 🧪 测试微调后的模型

训练完成后，使用测试脚本验证模型效果：

```bash
cd /workspace/hairong/code/Rex-Omni

# 测试特定checkpoint
python test_ovigod_model.py \
    --checkpoint finetuning/work_dirs/ovigod_sft/checkpoint-1500 \
    --image_path /workspace/hairong/data/ov-igod-dataset/sunrgbd_jpgs/1.jpg \
    --output_path test_result.jpg

# 测试其他图片
python test_ovigod_model.py \
    --checkpoint finetuning/work_dirs/ovigod_sft/checkpoint-1500 \
    --image_path /path/to/your/image.jpg \
    --output_path your_result.jpg
```

## 📋 训练参数说明

### 当前配置
- **数据集**: 6,701 个样本
- **训练轮数**: 3 epochs
- **Batch size**: 2 per GPU × 8 GPUs × 8 accumulation = 128（有效batch size）
- **学习率**: 
  - LLM: 2e-5
  - Vision Tower: 2e-6
- **保存频率**: 每 500 步保存一次
- **总训练步数**: 约 6701 / 128 × 3 = ~157 步

### 关键超参数
```bash
--num_train_epochs 3              # 训练轮数
--per_device_train_batch_size 2   # 每GPU的batch size
--gradient_accumulation_steps 8   # 梯度累积步数
--learning_rate 2e-5              # 主学习率
--vision_tower_lr 2e-6            # 视觉编码器学习率
--save_steps 500                  # 保存频率
--model_max_length 4096           # 最大序列长度
```

## 🎯 数据集特点

OV-IGOD 数据集使用**意图描述**（affordance）而非传统类别名：

**传统检测**:
- 类别: "bed", "lamp", "chair"

**OV-IGOD 检测**:
- 意图: "I long for a comfortable place to rest and rejuvenate after a long day"
- 意图: "I require soft lighting to create a calming atmosphere for reading before bed"

这使模型能够理解物体的**功能性**而非仅仅识别物体类别。

## ⚙️ 显存优化建议

### 如果遇到 OOM (Out of Memory) 错误：

**方案1: 减小batch size**
```bash
--per_device_train_batch_size 1
--gradient_accumulation_steps 16
```

**方案2: 使用 ZeRO-3 + Offload**
```bash
--deepspeed scripts/zero3_offload.json
```

**方案3: 冻结部分参数**
```bash
--tune_mm_vision False  # 冻结视觉编码器
```

**方案4: 减小图片分辨率**
编辑 `configs/sft_ovigod.py`:
```python
min_pixels = 16 * 28 * 28
max_pixels = 1280 * 28 * 28  # 从 2560 减小到 1280
```

## 📈 预期训练时间

基于 8×A100 80GB GPU：
- 单个epoch: ~5-10 分钟
- 总训练时间: ~15-30 分钟

实际时间取决于：
- GPU型号和数量
- Batch size设置
- 网络IO速度

## 🔧 常见问题

### Q1: 训练时显存不足
**A**: 参考上面的"显存优化建议"

### Q2: checkpoint在哪里？
**A**: `finetuning/work_dirs/ovigod_sft/checkpoint-XXX/`

### Q3: 如何选择最佳checkpoint？
**A**: 
- 查看训练日志的loss曲线
- 测试不同checkpoint的效果
- 通常最后几个checkpoint效果较好

### Q4: 可以继续训练吗？
**A**: 可以，在训练脚本中添加：
```bash
--resume_from_checkpoint work_dirs/ovigod_sft/checkpoint-XXX
```

## 📞 需要帮助？

如有问题，请检查：
1. ✅ 可视化结果是否正确
2. ✅ GPU显存是否足够
3. ✅ 训练日志中的错误信息
4. ✅ 数据路径是否正确

---

**生成时间**: 2025-11-15
**数据集**: OV-IGOD (6,701 samples)
**模型**: Rex-Omni (IDEA-Research)

