---
title: 视觉-语言模型
---

# 视觉-语言模型 (Vision-Language Models)

=== "中文"

    视觉-语言模型 (VLM) 利用大规模预训练的视觉-语言基础模型（如 CLIP、DINOv2）进行异常检测。
    核心优势是 **零样本 (zero-shot)** 或 **少样本 (few-shot)** 能力 -- 无需大量训练数据即可检测异常。

=== "English"

    Vision-Language Models (VLM) leverage large-scale pre-trained vision-language foundation models (e.g., CLIP, DINOv2) for anomaly detection.
    The core advantage is **zero-shot** or **few-shot** capability -- detecting anomalies without large training datasets.

---

## 快速对比

| 模型 | 注册名 | 基础模型 | 零样本 | 少样本 | 像素图 | 额外依赖 |
|:---|:---|:---|:---:|:---:|:---:|:---|
| WinCLIP | `winclip` / `vision_winclip` | OpenAI CLIP | 是 | 是 | 是 | `clip` |
| AnomalyDINO | `vision_anomalydino` | DINOv2 | 是* | 是 | 是 | `torch` |
| OpenCLIP PatchKNN | `vision_openclip_patch_map` | OpenCLIP | 否 | 是 | 是 | `open_clip` |
| PromptAD | `vision_promptad` | WideResNet50 | 否 | 是 | 否 | `torch` |

!!! note "零样本说明"
    AnomalyDINO 的零样本模式需要至少 1 张参考图进行阈值校准，但不需要传统意义上的"训练"。

---

## WinCLIP

=== "中文"

    WinCLIP (CVPR 2023) 利用 CLIP 的视觉-语言理解能力，通过滑动窗口注意力机制实现零样本和少样本异常检测。
    核心思路：用文本提示描述"正常"和"异常"，让 CLIP 判断图像区域属于哪一类。

=== "English"

    WinCLIP (CVPR 2023) leverages CLIP's vision-language understanding with sliding window attention for zero-shot and few-shot anomaly detection.
    Core idea: describe "normal" and "anomalous" with text prompts, and let CLIP judge which category each image region belongs to.

### 关键参数

| 参数 | 默认值 | 说明 |
|:---|:---|:---|
| `clip_model` | `"ViT-B/32"` | CLIP 模型架构 (`"RN50"`, `"RN101"`, `"ViT-B/32"`, `"ViT-L/14"`) |
| `window_size` | `224` | 滑动窗口大小 |
| `window_stride` | `112` | 滑动窗口步长 |
| `text_prompts` | `None` | 自定义文本提示（默认使用内置正常/异常提示） |
| `k_shot` | `0` | 少样本数量（0 = 零样本） |
| `scales` | `[1.0]` | 多尺度推理 |
| `device` | auto | 计算设备 |

### 零样本检测

```python
from pyimgano import create_model

# 零样本 -- 无需训练数据
model = create_model("winclip",
                     clip_model="ViT-B/32",
                     k_shot=0,
                     device="cuda")

# fit() 仅用于校准阈值，也可跳过直接调用 decision_function
model.fit(reference_images)

# 推理
scores = model.decision_function(test_images)
anomaly_map = model.get_anomaly_map(test_images[0])
```

### 少样本检测

```python
# 少样本 -- 仅需几张正常参考图
model = create_model("winclip",
                     clip_model="ViT-L/14",
                     k_shot=4,
                     device="cuda")
model.fit(few_normal_images)  # 4 张即可
scores = model.decision_function(test_images)
```

### 自定义文本提示

```python
custom_prompts = {
    "normal": [
        "a photo of a good product",
        "a clean surface without defects",
    ],
    "anomaly": [
        "a photo of a product with scratches",
        "a surface with cracks and damage",
    ],
}
model = create_model("winclip",
                     text_prompts=custom_prompts,
                     device="cuda")
```

---

## AnomalyDINO

=== "中文"

    AnomalyDINO (2025) 基于 DINOv2 的补丁嵌入 + kNN 检测器。DINOv2 的自监督预训练使其
    补丁特征天然具有强大的区分正常与异常的能力。支持少样本，推理快速。

=== "English"

    AnomalyDINO (2025) uses DINOv2 patch embeddings + kNN detector. DINOv2's self-supervised pre-training
    gives its patch features a natural ability to distinguish normal from anomalous. Supports few-shot, fast inference.

### 关键参数

| 参数 | 默认值 | 说明 |
|:---|:---|:---|
| `pretrained` | `False` | 是否通过 torch.hub 下载 DINOv2 权重 |
| `embedder` | `None` | 自定义补丁嵌入器（离线推荐） |
| `dino_model_name` | `"dinov2_vits14"` | DINOv2 模型变体 |
| `n_neighbors` | `1` | kNN 近邻数 |
| `coreset_sampling_ratio` | `1.0` | coreset 子采样比例 |
| `image_size` | `518` | 输入图像尺寸 |
| `aggregation_method` | `"topk_mean"` | 异常图聚合策略 |
| `device` | `"cpu"` | 计算设备 |

### 基本用法

```python
from pyimgano import create_model

model = create_model("vision_anomalydino",
                     pretrained=True,
                     n_neighbors=1,
                     device="cuda")

# 少量正常图即可训练 (建立 patch 记忆库)
model.fit(train_images)

# 推理
scores = model.decision_function(test_images)
anomaly_map = model.get_anomaly_map(test_images[0])
```

### 离线使用（自定义嵌入器）

=== "中文"

    生产环境中推荐使用自定义嵌入器，避免运行时下载权重。

=== "English"

    In production, use a custom embedder to avoid runtime weight downloads.

```python
from pyimgano import create_model
from pyimgano.models.anomalydino import TorchHubDinoV2Embedder

# 预先加载嵌入器
embedder = TorchHubDinoV2Embedder(
    model_name="dinov2_vits14",
    device="cuda",
    image_size=518,
)

model = create_model("vision_anomalydino",
                     embedder=embedder,
                     n_neighbors=1)
model.fit(train_images)
```

---

## OpenCLIP Patch Map

=== "中文"

    基于 OpenCLIP 的补丁模板距离异常图。学习正常补丁的"模板向量"（均值），
    推理时计算每个补丁到模板的余弦距离作为异常分数，输出像素级异常图。

=== "English"

    OpenCLIP-based patch template distance anomaly map. Learns a "template vector" (mean) of normal patches.
    At inference, computes cosine distance from each patch to the template as the anomaly score, outputting a pixel-level anomaly map.

### 关键参数

| 参数 | 默认值 | 说明 |
|:---|:---|:---|
| `openclip_model_name` | `"ViT-B-32"` | OpenCLIP 模型名称 |
| `openclip_pretrained` | `None` | 预训练权重标识 |
| `normalize_embeddings` | `True` | 是否 L2 归一化嵌入 |
| `aggregation_method` | `"topk_mean"` | 异常图聚合策略 |
| `device` | `"cpu"` | 计算设备 |

```python
from pyimgano import create_model

model = create_model("vision_openclip_patch_map",
                     openclip_model_name="ViT-B-32",
                     device="cuda")
model.fit(train_images)

scores = model.decision_function(test_images)
anomaly_map = model.get_anomaly_map(test_images[0])
```

---

## PromptAD

=== "中文"

    PromptAD (CVPR 2024) 通过仅从正常样本学习视觉提示 (visual prompts) 实现少样本异常检测。
    核心思路：学习一组可训练的提示向量，引导预训练特征提取器关注正常模式。

=== "English"

    PromptAD (CVPR 2024) achieves few-shot anomaly detection by learning visual prompts from only normal samples.
    Core idea: learn a set of trainable prompt vectors to guide the pre-trained feature extractor to focus on normal patterns.

### 关键参数

| 参数 | 默认值 | 说明 |
|:---|:---|:---|
| `backbone` | `"wide_resnet50"` | 特征提取主干 |
| `num_prompts` | `10` | 可学习提示数量 |
| `prompt_dim` | `512` | 提示向量维度 |
| `context_length` | `16` | 上下文长度 |
| `learning_rate` | `1e-3` | 学习率 |
| `epochs` | `30` | 训练轮数 |
| `device` | `"cuda"` | 计算设备 |

```python
from pyimgano import create_model

model = create_model("vision_promptad",
                     num_prompts=10,
                     epochs=30,
                     device="cuda")
model.fit(few_normal_images)
scores = model.decision_function(test_images)
```

---

## 安装依赖

=== "中文"

    VLM 模型需要额外依赖。根据所用模型安装对应的扩展包。

=== "English"

    VLM models require extra dependencies. Install the appropriate extras based on the model you use.

```bash
# WinCLIP -- OpenAI CLIP
pip install pyimgano[clip]

# OpenCLIP Patch Map -- open_clip
pip install pyimgano[clip]

# AnomalyDINO -- torch + torchvision (DINOv2 via torch.hub)
pip install pyimgano[torch]

# PromptAD -- torch + torchvision
pip install pyimgano[torch]

# 安装所有 VLM 依赖
pip install pyimgano[clip,torch]
```

---

## 选择建议

!!! tip "何时选择 VLM"

    === "中文"

        - **零样本**: 无训练数据或快速原型 -> `winclip` (k_shot=0)
        - **少量参考图 (1-10 张)**: -> `vision_anomalydino` 或 `winclip` (k_shot>0)
        - **需要像素定位 + 无训练**: -> `vision_anomalydino`
        - **自定义语义描述**: -> `winclip` (自定义 text_prompts)
        - **CVPR 2024 SOTA 少样本**: -> `vision_promptad`

    === "English"

        - **Zero-shot**: no training data or rapid prototyping -> `winclip` (k_shot=0)
        - **Few reference images (1-10)**: -> `vision_anomalydino` or `winclip` (k_shot>0)
        - **Pixel localization + no training**: -> `vision_anomalydino`
        - **Custom semantic descriptions**: -> `winclip` (custom text_prompts)
        - **CVPR 2024 SOTA few-shot**: -> `vision_promptad`
