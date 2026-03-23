# 评估结果输出格式说明

## 📄 输出文件示例

两个评估脚本都会生成包含详细预测信息的JSON文件：
- `evaluate_three_datasets.sh` → `evaluation_three_datasets_results_sft.json`
- `evaluate_three_datasets_ensemble.sh` → `evaluation_three_datasets_results_ensemble.json`

---

## 🔍 JSON结构详解

### 1. **标准评估输出** (`evaluate_three_datasets.py`)

```json
{
  "checkpoint": "/path/to/model/checkpoint",
  "dataset_results": {
    "coco_outdoor": {
      "num_samples": 1000,
      "mean_iou": 0.6053,
      "AP@50": 0.4769,
      "AP@75": 0.2947,
      "AP@50:95": 0.2877
    },
    "scannet": { ... },
    "egoobject": { ... }
  },
  "detailed_predictions": {
    "coco_outdoor": {
      "sample_001": {
        "ground_truth": [
          [100.0, 200.0, 300.0, 400.0]  // GT bbox: [x0, y0, x1, y1]
        ],
        "predictions": [
          [105.2, 198.5, 302.1, 405.3],  // 预测bbox 1
          [500.0, 600.0, 700.0, 800.0]   // 预测bbox 2 (可能是噪声)
        ],
        "iou": 0.87  // ⭐ 新增：该样本的最佳IoU
      },
      "sample_002": { ... },
      ...
    },
    "scannet": { ... },
    "egoobject": { ... }
  }
}
```

#### 🔑 关键字段

| 字段 | 说明 | 示例值 |
|------|------|--------|
| `checkpoint` | 评估使用的模型路径 | `"finetuning/work_dirs/intention_sft"` |
| `dataset_results` | 每个数据集的汇总指标 | 见下方 |
| `detailed_predictions` | 每个样本的详细预测 | 见下方 |

---

### 2. **Ensemble评估输出** (`evaluate_three_datasets_ensemble.py`)

```json
{
  "checkpoint": "/path/to/model/checkpoint",
  "evaluation_type": "three_datasets_ensemble",
  "ensemble_config": {
    "n_samples": 5,           // 每个样本推理5次
    "vote_threshold": 0.4,    // 投票阈值 (保留出现 ≥40%次数的框)
    "iou_threshold": 0.5,     // 聚类阈值
    "temperature": 1.0,       // 采样温度
    "batch_size": 8           // 批大小
  },
  "per_dataset_metrics": {
    "coco_outdoor": {
      "mean_iou": 0.6123,
      "AP@50": 0.4850,
      "AP@75": 0.3010,
      "AP@50:95": 0.2920,
      "ensemble_stats": {
        "avg_boxes_before_vote": 8.5,  // 投票前平均框数 (5次×~1.7框/次)
        "avg_boxes_after_vote": 1.2,   // 投票后平均框数
        "filtering_ratio": 0.14         // 过滤比例 (保留14%)
      }
    },
    "scannet": { ... },
    "egoobject": { ... }
  },
  "detailed_predictions": {
    "coco_outdoor": {
      "sample_001": {
        "ground_truth": [
          [100.0, 200.0, 300.0, 400.0]
        ],
        "predictions": [
          {
            "coords": [102.5, 199.8, 301.2, 402.1],  // Ensemble后的平均坐标
            "confidence": 0.85,                       // 平均置信度
            "vote_count": 5                           // 该簇的投票数 (出现了5次)
          },
          {
            "coords": [120.0, 220.0, 320.0, 420.0],
            "confidence": 0.72,
            "vote_count": 3                           // 只出现3次
          }
        ],
        "iou": 0.89  // ⭐ 新增：该样本的最佳IoU
      },
      "sample_002": { ... },
      ...
    },
    "scannet": { ... },
    "egoobject": { ... }
  }
}
```

#### 🆚 与标准评估的区别

| 特性 | 标准评估 | Ensemble评估 |
|------|----------|-------------|
| **预测格式** | 简单列表 `[[x0,y0,x1,y1]]` | 带元信息 `{coords, confidence, vote_count}` |
| **坐标来源** | 单次推理 | 多次推理的聚类+平均 |
| **IoU保存** | ✅ 每个样本的最佳IoU | ✅ 每个样本的最佳IoU |
| **额外统计** | ❌ | ✅ `ensemble_stats` (投票前后框数) |

---

## 📊 数据集指标字段

### `dataset_results` / `per_dataset_metrics`

```json
{
  "num_samples": 1000,        // 测试样本数
  "mean_iou": 0.6053,         // 平均IoU (所有样本的最高IoU平均值)
  "AP@50": 0.4769,            // IoU≥0.5时的准确率
  "AP@75": 0.2947,            // IoU≥0.75时的准确率
  "AP@50:95": 0.2877          // IoU从0.5到0.95的平均AP
}
```

---

## 💡 如何使用这些数据

### 1️⃣ **查看整体指标**

```python
import json

with open("evaluation_three_datasets_results_sft.json") as f:
    data = json.load(f)

# 打印汇总表格
print("Dataset       Mean IoU   AP@50    AP@75    AP@50:95")
for dataset, metrics in data['dataset_results'].items():
    print(f"{dataset:<12} {metrics['mean_iou']:.4f}   {metrics['AP@50']:.4f}   {metrics['AP@75']:.4f}   {metrics['AP@50:95']:.4f}")
```

### 2️⃣ **分析单个样本的预测**

```python
# 查看某个失败样本的预测（新版：直接读取保存的IoU）
predictions = data['detailed_predictions']['coco_outdoor']

# 快速筛选低IoU样本
low_iou_samples = []
for sample_id, pred_data in predictions.items():
    iou = pred_data['iou']  # ⭐ 直接读取保存的IoU，无需重新计算
    if iou < 0.3:
        low_iou_samples.append((sample_id, iou, pred_data))

# 按IoU排序，找出最差的样本
low_iou_samples.sort(key=lambda x: x[1])

print(f"找到 {len(low_iou_samples)} 个低IoU样本")
for sample_id, iou, pred_data in low_iou_samples[:10]:  # 显示最差的10个
    print(f"❌ {sample_id}: IoU={iou:.3f}")
    print(f"   GT:   {pred_data['ground_truth'][0]}")
    if len(pred_data['predictions']) > 0:
        print(f"   Pred: {pred_data['predictions'][0]}")
    else:
        print(f"   Pred: 没有预测")
```

### 3️⃣ **对比标准 vs Ensemble**

```python
# 加载两个结果
with open("evaluation_three_datasets_results_sft.json") as f:
    standard = json.load(f)
    
with open("evaluation_three_datasets_results_ensemble.json") as f:
    ensemble = json.load(f)

# 对比AP提升
for dataset in ['coco_outdoor', 'scannet', 'egoobject']:
    ap50_std = standard['dataset_results'][dataset]['AP@50']
    ap50_ens = ensemble['per_dataset_metrics'][dataset]['AP@50']
    improvement = (ap50_ens - ap50_std) / ap50_std * 100
    
    print(f"{dataset}: {ap50_std:.4f} → {ap50_ens:.4f} (+{improvement:.1f}%)")
```

### 4️⃣ **可视化预测框**

```python
from PIL import Image, ImageDraw

def visualize_prediction(image_path, sample_id, pred_data):
    """可视化某个样本的预测结果"""
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    
    # 画GT (绿色)
    gt = pred_data['ground_truth'][0]
    draw.rectangle(gt, outline='green', width=3)
    
    # 画预测 (红色)
    if isinstance(pred_data['predictions'][0], dict):
        # Ensemble格式
        pred = pred_data['predictions'][0]['coords']
    else:
        # 标准格式
        pred = pred_data['predictions'][0]
    
    draw.rectangle(pred, outline='red', width=3)
    
    img.save(f"vis_{sample_id}.jpg")
```

---

## 🎯 关键观察点

### ✅ **好的指标**
- `mean_iou > 0.6` → 整体定位准确
- `AP@50 > 0.5` → 半数样本IoU超过0.5
- `AP@75 > 0.3` → 30%样本达到高精度定位

### ⚠️ **需要关注**
- `mean_iou` 高但 `AP@50` 低 → 可能有噪声预测拉低了准确率
- `AP@75` / `AP@50` < 0.6 → 预测框不够精细
- Ensemble的 `avg_boxes_after_vote` > 2 → 投票阈值太低，没有充分过滤

### 🔥 **Ensemble改善效果**
- `AP@50` 提升 > 2% → Ensemble有效
- `filtering_ratio` < 0.2 → 成功过滤80%的噪声框
- `vote_count` 高的框通常更准确

---

## 📦 文件位置

```
IntentionDetection/
├── evaluation_three_datasets_results_sft.json       # 标准评估结果
├── evaluation_three_datasets_results_ensemble.json  # Ensemble评估结果
└── EVALUATION_OUTPUT_FORMAT.md                      # 本文档
```

---

**提示**：JSON文件可能很大（如果包含所有样本），建议用 `jq` 或 Python 脚本按需查询，而不是直接打开整个文件。

```bash
# 只查看汇总指标
jq '.dataset_results' evaluation_three_datasets_results_sft.json

# 查看某个样本的预测和IoU
jq '.detailed_predictions.coco_outdoor.sample_001' evaluation_three_datasets_results_sft.json

# 快速找出IoU < 0.3的样本数量
jq '[.detailed_predictions.coco_outdoor | to_entries[] | select(.value.iou < 0.3)] | length' evaluation_three_datasets_results_sft.json

# 找出IoU最低的5个样本
jq '[.detailed_predictions.coco_outdoor | to_entries[] | {id: .key, iou: .value.iou}] | sort_by(.iou) | .[0:5]' evaluation_three_datasets_results_sft.json
```

