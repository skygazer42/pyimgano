---
title: 深度学习模型
---

# 深度学习模型 (Deep Learning Models)

=== "中文"

    深度学习模型利用预训练网络提取视觉特征，在工业缺陷检测中实现最先进精度。
    大多支持像素级异常定位（anomaly map）。推荐使用 GPU 加速训练和推理。

=== "English"

    Deep learning models leverage pre-trained networks for visual feature extraction, achieving state-of-the-art accuracy in industrial defect detection.
    Most support pixel-level anomaly localization (anomaly maps). GPU recommended for training and inference.

---

## 快速对比

| 算法 | 年份 | 注册名 | 速度 | 精度 | 显存 | 训练时间 |
|:---|:---:|:---|:---:|:---:|:---:|:---:|
| PatchCore | 2022 | `vision_patchcore` | 中 | 极高 | 中 | 无需训练 |
| SoftPatch | 2022 | `vision_softpatch` | 中 | 极高 | 中 | 无需训练 |
| SimpleNet | 2023 | `vision_simplenet` | 极快 | 极高 | 低 | 数分钟 |
| STFPM | 2021 | `vision_stfpm` | 快 | 高 | 低 | ~30 min |
| EfficientAD | 2024 | `efficient_ad` | 极快 | 极高 | 低 | ~10 min |
| ReverseDistillation | 2022 | `vision_reverse_distillation` | 快 | 高 | 低 | ~20 min |
| FastFlow | 2021 | `vision_fastflow` | 快 | 高 | 中 | ~30 min |
| CFlow | 2022 | `vision_cflow` | 中 | 高 | 中 | ~30 min |
| DRAEM | 2021 | `vision_draem` | 中 | 高 | 中 | ~1 hr |
| CutPaste | 2021 | `vision_cutpaste` | 中 | 中高 | 中 | ~1 hr |
| AnomalyDINO | 2025 | `vision_anomalydino` | 中 | 极高 | 中 | 无需训练 |

---

## 记忆库方法 (Memory Bank)

=== "中文"

    记忆库方法不需要反向传播训练。它们提取正常样本的补丁特征建立"记忆库"，
    推理时通过最近邻搜索计算异常分数。

=== "English"

    Memory bank methods require no backpropagation training. They extract patch features from normal samples to build a "memory bank" and compute anomaly scores via nearest neighbor search during inference.

### PatchCore

!!! info "SOTA -- CVPR 2022"
    PatchCore 是工业异常检测领域引用最多的方法之一。通过 coreset 子采样实现高效存储。

```python
from pyimgano import create_model

model = create_model("vision_patchcore",
                     backbone="wide_resnet50",
                     coreset_sampling_ratio=0.1,
                     n_neighbors=9,
                     device="cuda")
model.fit(train_images)

# 图像级分数
scores = model.decision_function(test_images)

# 像素级异常图
anomaly_map = model.get_anomaly_map(test_images[0])
```

**关键参数：**

| 参数 | 默认值 | 说明 |
|:---|:---|:---|
| `backbone` | `wide_resnet50` | 特征主干网络 |
| `coreset_sampling_ratio` | `0.1` | coreset 保留比例 (0.0-1.0) |
| `n_neighbors` | `9` | k 近邻数 |
| `layers` | `['layer2', 'layer3']` | 特征提取层 |
| `device` | `cpu` | 计算设备 |

### SoftPatch

=== "中文"

    SoftPatch 在 PatchCore 基础上引入鲁棒性机制，对训练数据中的噪声样本更不敏感。
    使用 DINOv2 提取补丁嵌入，支持少样本场景。

=== "English"

    SoftPatch introduces robustness mechanisms on top of PatchCore, making it less sensitive to noisy training samples.
    Uses DINOv2 for patch embeddings, supports few-shot scenarios.

```python
model = create_model("vision_softpatch",
                     pretrained=True,
                     n_neighbors=1,
                     coreset_sampling_ratio=1.0,
                     device="cuda")
```

---

## 师生蒸馏方法 (Student-Teacher)

=== "中文"

    使用冻结的预训练教师网络指导可训练的学生网络。正常区域学生能模仿教师输出，
    异常区域产生差异，差异即为异常信号。

=== "English"

    A frozen pre-trained teacher network guides a trainable student network. The student mimics the teacher on normal regions; differences on anomalous regions serve as the anomaly signal.

### STFPM

```python
model = create_model("vision_stfpm",
                     backbone="resnet18",
                     epochs=100,
                     batch_size=32,
                     device="cuda")
model.fit(train_images)
scores = model.decision_function(test_images)
anomaly_map = model.get_anomaly_map(test_images[0])
```

### EfficientAD

=== "中文"

    毫秒级推理延迟，适合实时工业部署。轻量学生-教师蒸馏架构。

=== "English"

    Millisecond-level inference latency, suitable for real-time industrial deployment. Lightweight student-teacher distillation architecture.

```python
model = create_model("efficient_ad",
                     epochs=10,
                     batch_size=16,
                     image_size=256,
                     device="cuda")
model.fit(train_images)
```

### Reverse Distillation

```python
model = create_model("vision_reverse_distillation",
                     backbone="resnet18",
                     epoch_num=20,
                     device="cuda")
model.fit(train_images)
```

---

## 流模型方法 (Flow-based)

=== "中文"

    使用归一化流将正常特征分布建模为标准正态分布。异常样本的对数似然偏低。

=== "English"

    Use normalizing flows to model the normal feature distribution as standard normal. Anomalous samples have lower log-likelihood.

### FastFlow

```python
model = create_model("vision_fastflow",
                     backbone="resnet18",
                     n_flow_steps=8,
                     epoch_num=20,
                     device="cuda")
model.fit(train_images)
```

### CFlow

```python
model = create_model("vision_cflow",
                     backbone="resnet18",
                     n_flows=8,
                     epochs=50,
                     device="cuda")
model.fit(train_images)
```

---

## 重建方法 (Reconstruction)

=== "中文"

    训练网络重建正常图像。异常区域无法被良好重建，重建误差即为异常信号。

=== "English"

    Train networks to reconstruct normal images. Anomalous regions cannot be well reconstructed; reconstruction error serves as the anomaly signal.

### DRAEM

```python
model = create_model("vision_draem",
                     image_size=256,
                     epochs=100,
                     batch_size=8,
                     device="cuda")
model.fit(train_images)
anomaly_map = model.get_anomaly_map(test_images[0])
```

---

## 自监督方法 (Self-supervised)

### CutPaste

=== "中文"

    通过合成"剪贴"增强创建伪异常样本，训练网络区分正常与伪异常。

=== "English"

    Creates pseudo-anomaly samples via synthetic "cut-paste" augmentation and trains the network to distinguish normal from pseudo-anomalous.

```python
model = create_model("vision_cutpaste",
                     backbone="resnet18",
                     augment_type="3way",
                     epochs=50,
                     device="cuda")
model.fit(train_images)
```

---

## 基础模型方法 (Foundation Models)

### SimpleNet

!!! info "CVPR 2023 -- 极快推理"
    仅 1M 额外参数，训练快，推理速度接近实时。

```python
model = create_model("vision_simplenet",
                     backbone="wide_resnet50",
                     device="cuda")
model.fit(train_images)
```

### AnomalyDINO

=== "中文"

    基于 DINOv2 的补丁 kNN 检测器，支持少样本和零样本。详见 [VLM 模型页](vlm.md)。

=== "English"

    DINOv2-based patch kNN detector, supports few-shot and zero-shot. See [VLM Models page](vlm.md).

---

## 检测器接口协议 (Detector Contract)

=== "中文"

    所有深度模型遵循统一的 sklearn 风格接口。

=== "English"

    All deep models follow a unified sklearn-style interface.

```python
# 训练
model.fit(X_train)

# 图像级异常分数（越高越异常）
scores = model.decision_function(X_test)

# 二值预测 (1=异常, 0=正常)
predictions = model.predict(X_test)

# 像素级异常图 (如果模型支持)
anomaly_map = model.get_anomaly_map(image)  # shape: (H, W), float32
```

!!! note "分数约定"
    所有模型遵循 **分数越高 = 越异常** 的约定。`predict()` 使用 `contamination` 参数自动计算阈值。

---

## 自定义模型注册

=== "中文"

    通过 `@register_model` 装饰器将自定义检测器注册到 pyimgano 模型库。

=== "English"

    Register custom detectors to the pyimgano model registry via the `@register_model` decorator.

```python
from pyimgano.models.registry import register_model

@register_model(
    "my_custom_detector",
    tags=("vision", "deep", "custom", "pixel_map"),
    metadata={
        "description": "My custom anomaly detector",
        "paper": "...",
        "year": 2025,
    },
)
class MyCustomDetector:
    def __init__(self, *, contamination=0.1, device="cpu", **kwargs):
        self.contamination = contamination
        self.device = device

    def fit(self, x, y=None):
        # 训练逻辑
        return self

    def decision_function(self, x):
        # 返回异常分数 (N,)
        ...

    def predict(self, x):
        # 返回二值预测 (N,)
        ...

    def get_anomaly_map(self, image):
        # 返回像素级异常图 (H, W)
        ...
```

=== "中文"

    注册后即可通过 `create_model("my_custom_detector")` 使用。

=== "English"

    After registration, use it via `create_model("my_custom_detector")`.

---

## Anomalib 检查点封装

=== "中文"

    pyimgano 提供 anomalib 检查点的封装器，可直接加载 anomalib 训练的模型用于推理。
    需要安装 `pyimgano[anomalib]` 扩展。

=== "English"

    pyimgano provides wrappers for anomalib checkpoints, allowing direct loading of anomalib-trained models for inference.
    Requires the `pyimgano[anomalib]` extra.

```bash
pip install pyimgano[anomalib]
```

```python
from pyimgano import create_model

model = create_model("vision_anomalib_checkpoint",
                     checkpoint_path="/path/to/model.ckpt",
                     device="cuda")
model.fit(reference_images)  # 设定阈值

scores = model.decision_function(test_images)
anomaly_map = model.get_anomaly_map(test_images[0])
```

=== "中文"

    可用的 anomalib 后端模型包括：

=== "English"

    Available anomalib backend models include:

| 注册名 | 对应算法 |
|:---|:---|
| `vision_patchcore_anomalib` | PatchCore |
| `vision_padim_anomalib` | PaDiM |
| `vision_stfpm_anomalib` | STFPM |
| `vision_draem_anomalib` | DRAEM |
| `vision_fastflow_anomalib` | FastFlow |
| `vision_reverse_distillation_anomalib` | Reverse Distillation |
| `vision_efficientad_anomalib` | EfficientAD |
| `vision_cflow_anomalib` | CFlow |
| `vision_dfm_anomalib` | DFM |
| `vision_dinomaly_anomalib` | Dinomaly |
