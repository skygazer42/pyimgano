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

| 模型 | 注册名 | 关系 | 基础模型 | 像素图 | 额外依赖 |
|:---|:---|:---|:---|:---:|:---|
| WinCLIP crop proxy | `winclip` / `vision_winclip` | `inspired` | OpenAI CLIP | 是 | `clip` |
| WinCLIP upstream | `vision_winclip_anomalib` | `external-backend` | anomalib | 是 | `anomalib` |
| AnomalyDINO-style kNN | `vision_anomalydino` | `inspired` | DINOv2 | 是 | `torch` |
| OpenCLIP PatchKNN | `vision_openclip_patch_map` | `not-applicable` | OpenCLIP | 是 | `open_clip` |
| PromptAD visual proxy | `vision_promptad` | `inspired` | WideResNet50 | 否 | `torch` |

!!! note "零样本说明"
    AnomalyDINO 的零样本模式需要至少 1 张参考图进行阈值校准，但不需要传统意义上的"训练"。

!!! warning "论文复现状态"
    本页的本地 `vision_winclip`、`vision_anomalydino` 和 `vision_promptad` 是实验代理，
    不是对应论文的完整实现。论文关系以 `model_info(...)["metadata"]["paper_fidelity"]`
    为准；WinCLIP 的上游实现路径是 `vision_winclip_anomalib`。

---

## WinCLIP-related crop proxy

=== "中文"

    本地实现对裁剪窗口分别做 CLIP 编码和文本打分，只保留了 WinCLIP 的概念动机。
    它没有实现论文的 token 级多尺度窗口嵌入与聚合，因此不能作为论文复现。

=== "English"

    The local implementation scores cropped windows with CLIP and text prompts.
    It omits the paper's token-level multi-scale window embeddings and aggregation,
    so it is an experimental proxy rather than a WinCLIP reproduction.

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

## AnomalyDINO-style kNN proxy

=== "中文"

    本地实现是 DINOv2 补丁嵌入 + kNN 基线，适合少样本实验，但没有复现
    AnomalyDINO 论文的完整方法。

=== "English"

    The local implementation is a DINOv2 patch-embedding + kNN baseline. It is
    useful for few-shot experiments but does not reproduce the full AnomalyDINO method.

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

## PromptAD-related visual proxy

=== "中文"

    本地实现是 WideResNet 特征适配器。它没有实现论文中的 CLIP 文本提示、语义拼接和
    显式异常间隔损失，因此仅作为实验代理保留。

=== "English"

    The local implementation is a WideResNet feature adapter. It omits the
    paper's CLIP text prompts, semantic concatenation, and explicit anomaly-margin
    loss, so it is retained only as an experimental proxy.

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

        - **论文级 WinCLIP 路径**: -> `vision_winclip_anomalib`
        - **零样本实验代理**: -> `winclip` (k_shot=0)
        - **少量参考图 (1-10 张) 的 kNN 基线**: -> `vision_anomalydino`
        - **需要像素定位 + 无训练**: -> `vision_anomalydino`
        - **自定义语义描述**: -> `winclip` (自定义 text_prompts)
        - **PromptAD 概念实验**: -> `vision_promptad`（非论文复现）

    === "English"

        - **Paper-backed WinCLIP path**: -> `vision_winclip_anomalib`
        - **Zero-shot experimental proxy**: -> `winclip` (k_shot=0)
        - **Few-reference kNN baseline (1-10)**: -> `vision_anomalydino`
        - **Pixel localization + no training**: -> `vision_anomalydino`
        - **Custom semantic descriptions**: -> `winclip` (custom text_prompts)
        - **PromptAD concept experiment**: -> `vision_promptad` (not a reproduction)
