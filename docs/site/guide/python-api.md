---
title: Python API
---

# Python API

=== "中文"

    pyimgano 的 Python API 遵循 scikit-learn 风格的四步法：创建 → 训练 → 评分 → 预测。支持 120+ 种模型，涵盖图像级和像素级异常检测。

=== "English"

    The pyimgano Python API follows a scikit-learn-style four-step workflow: create → fit → score → predict. It supports 120+ models covering both image-level and pixel-level anomaly detection.

---

## 核心工作流

### 1. 创建模型

```python
from pyimgano import create_model

# 经典模型 (无需 GPU)
model = create_model("vision_iforest")

# 深度模型 (需要 torch)
model = create_model("vision_patchcore", feature_extractor="resnet18")

# 带参数创建
model = create_model("vision_ecod", contamination=0.05)
```

=== "中文"

    `create_model()` 是统一入口，接受模型注册名和任意关键字参数。运行 `pyim --list models` 查看全部可用模型。

=== "English"

    `create_model()` is the unified entry point, accepting a registered model name and arbitrary keyword arguments. Run `pyim --list models` to see all available models.

### 2. 训练

```python
import numpy as np

# 准备训练数据: 仅正常样本, shape (N, H, W, C), dtype uint8
X_train = np.random.randint(0, 255, (100, 128, 128, 3), dtype=np.uint8)

# 训练 (仅需正常样本)
model.fit(X_train)
```

!!! info "训练数据要求"

    - 仅需 **正常样本**（无异常标注）
    - 推荐格式：`(N, H, W, C)` uint8 NumPy 数组
    - 最少样本数取决于模型：经典模型 ~10 张，深度模型 ~30 张

### 3. 异常评分

```python
X_test = np.random.randint(0, 255, (20, 128, 128, 3), dtype=np.uint8)

# 返回每张图像的异常分数 (越高越异常)
scores = model.decision_function(X_test)
# scores.shape: (20,)
```

### 4. 二值预测

```python
# 返回二值标签: 1=异常, 0=正常
predictions = model.predict(X_test)
# predictions.shape: (20,)
```

---

## 像素级异常检测

=== "中文"

    像素级模型输出与输入图像同尺寸的异常热图，可精确定位缺陷区域。

=== "English"

    Pixel-level models output anomaly heatmaps with the same dimensions as the input image, enabling precise defect localization.

### 获取异常热图

```python
from pyimgano import create_model

model = create_model("vision_patchcore", feature_extractor="resnet18")
model.fit(X_train)

# 单张图像 → 异常热图
anomaly_map = model.get_anomaly_map(X_test[0])
# anomaly_map.shape: (H, W)

# 批量 → 像素级异常分数
pixel_scores = model.decision_function(X_test)
# pixel_scores.shape: (N, H, W) 对于像素级模型
```

### 像素级二值预测

```python
# 批量像素级预测
pixel_predictions = model.predict_anomaly_map(X_test)
# pixel_predictions.shape: (N, H, W), 值为 0 或 1
```

---

## 经典模型 vs 深度模型

```python
# === 经典模型 ===
# 无需 GPU，依赖少，速度快
model_iforest = create_model("vision_iforest")
model_ecod = create_model("vision_ecod")

# === 深度模型 ===
# 需要 pip install "pyimgano[torch]"
model_patchcore = create_model("vision_patchcore",
    feature_extractor="resnet18"
)
model_padim = create_model("vision_padim",
    feature_extractor="wide_resnet50_2"
)
```

=== "中文"

    | 类型 | 代表模型 | GPU | 额外依赖 | 适用场景 |
    |------|---------|-----|---------|---------|
    | 经典 | `vision_iforest`, `vision_ecod` | 否 | 无 | 快速原型、资源受限 |
    | 深度 | `vision_patchcore`, `vision_padim` | 推荐 | `torch` | 高精度、像素级定位 |

=== "English"

    | Type | Examples | GPU | Extras | Use Case |
    |------|---------|-----|--------|----------|
    | Classical | `vision_iforest`, `vision_ecod` | No | None | Fast prototyping, resource-constrained |
    | Deep | `vision_patchcore`, `vision_padim` | Recommended | `torch` | High accuracy, pixel-level localization |

---

## 特征提取器

=== "中文"

    深度模型通过 `feature_extractor` 参数选择骨干网络。不同提取器在精度和速度之间有不同权衡。

=== "English"

    Deep models select a backbone network via the `feature_extractor` parameter. Different extractors offer varying trade-offs between accuracy and speed.

```python
# 轻量级 (快速，适合边缘部署)
model = create_model("vision_patchcore", feature_extractor="resnet18")

# 标准 (精度与速度平衡)
model = create_model("vision_patchcore", feature_extractor="wide_resnet50_2")

# 高精度 (更慢，更精确)
model = create_model("vision_padim", feature_extractor="wide_resnet50_2")
```

!!! tip "选择建议"

    - 初始实验使用 `resnet18`（快速迭代）
    - 生产部署根据精度需求选择 `wide_resnet50_2` 或更大骨干网络
    - 使用 `pyimgano-benchmark` 系统对比不同提取器的效果

---

## 评估指标

```python
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

# 假设有真实标签
y_true = np.array([0, 0, 0, 1, 1])

# 使用 decision_function 的分数计算指标
scores = model.decision_function(X_test)
auroc = roc_auc_score(y_true, scores)
ap = average_precision_score(y_true, scores)

# 使用 predict 的标签计算 F1
predictions = model.predict(X_test)
f1 = f1_score(y_true, predictions)

print(f"AUROC: {auroc:.4f}, AP: {ap:.4f}, F1: {f1:.4f}")
```

=== "中文"

    pyimgano 兼容 scikit-learn 指标体系。常用指标：

    - **AUROC** — 区分正常/异常的整体能力
    - **AP (Average Precision)** — 异常样本稀少时更稳健
    - **F1** — 精确率与召回率的调和均值

=== "English"

    pyimgano is compatible with the scikit-learn metrics ecosystem. Common metrics:

    - **AUROC** — Overall ability to distinguish normal/anomalous
    - **AP (Average Precision)** — More robust when anomalies are rare
    - **F1** — Harmonic mean of precision and recall

---

## 完整示例

```python
import numpy as np
from pyimgano import create_model
from sklearn.metrics import roc_auc_score

# 1. 准备数据
X_train = np.random.randint(0, 255, (50, 64, 64, 3), dtype=np.uint8)
X_test = np.random.randint(0, 255, (10, 64, 64, 3), dtype=np.uint8)
y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

# 2. 创建并训练模型
model = create_model("vision_iforest")
model.fit(X_train)

# 3. 推理
scores = model.decision_function(X_test)
predictions = model.predict(X_test)

# 4. 评估
auroc = roc_auc_score(y_true, scores)
print(f"AUROC: {auroc:.4f}")
print(f"Predictions: {predictions}")
```

---

## 下一步

- [CLI 概览](cli-overview.md) — 命令行工具链完整参考
- [训练](training.md) — 使用配置文件进行高级训练
- [推理](inference.md) — 批量推理与生产集成
