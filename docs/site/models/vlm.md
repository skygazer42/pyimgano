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
| PromptAD | `vision_promptad` | `paper-adaptation` | VV-CLIP ViT-B/16+ | 是 | `open_clip` |

!!! note "零样本说明"
    AnomalyDINO 的零样本模式需要至少 1 张参考图进行阈值校准，但不需要传统意义上的"训练"。

!!! warning "论文复现状态"
    本页的本地 `vision_winclip` 和 `vision_anomalydino` 是实验代理，
    不是对应论文的完整实现。论文关系以 `model_info(...)["metadata"]["paper_fidelity"]`
    为准；WinCLIP 的上游实现路径是 `vision_winclip_anomalib`。`vision_promptad`
    是论文适配实现，但论文的图像级与像素级提示需分别训练。

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

## PromptAD

=== "中文"

    本地实现使用论文的冻结 LAION-400M ViT-B/16+、VV-attention、语义拼接、
    零间隔 EAM、MAP/LAP 分布对齐、双层正常视觉记忆和调和融合。分类与分割
    对应作者的两套训练脚本，应分别拟合模型。

=== "English"

    The local implementation follows the frozen LAION-400M ViT-B/16+,
    V-V attention, semantic concatenation, zero-margin EAM, MAP/LAP alignment,
    two-layer normal visual memory, and harmonic fusion. Classification and
    segmentation prompts are trained separately, matching the authors' scripts.

### 关键参数

| 参数 | 默认值 | 说明 |
|:---|:---|:---|
| `class_name` | `"object"` | 被检测对象名称；复现实验时传数据集类别 |
| `openclip_model_name` | `"ViT-B-16-plus-240"` | 论文主干 |
| `openclip_pretrained` | `"laion400m_e32"` | 论文预训练权重 |
| `n_ctx` / `n_ctx_ab` | `4` / `1` | 正常前缀 / 可学习异常后缀长度 |
| `n_pro` / `n_pro_ab` | `1` / `4` | 正常提示 / 可学习异常后缀数量 |
| `learning_rate` | `0.002` | SGD 学习率 |
| `epochs` | `100` | 官方训练脚本轮数 |
| `training_task` | `"classification"` | `classification` 或 `segmentation` |
| `device` | `"cuda"` | 计算设备 |

```python
from pyimgano import create_model

image_model = create_model("vision_promptad",
                           class_name="carpet",
                           training_task="classification",
                           device="cuda")
image_model.fit(few_normal_images)
scores = image_model.decision_function(test_images)

pixel_model = create_model("vision_promptad",
                           class_name="carpet",
                           training_task="segmentation",
                           device="cuda")
pixel_model.fit(few_normal_images)
maps = pixel_model.predict_anomaly_map(test_images)
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

# PromptAD -- OpenCLIP
pip install pyimgano[clip]

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
        - **PromptAD 少样本论文路径**: -> `vision_promptad`

    === "English"

        - **Paper-backed WinCLIP path**: -> `vision_winclip_anomalib`
        - **Zero-shot experimental proxy**: -> `winclip` (k_shot=0)
        - **Few-reference kNN baseline (1-10)**: -> `vision_anomalydino`
        - **Pixel localization + no training**: -> `vision_anomalydino`
        - **Custom semantic descriptions**: -> `winclip` (custom text_prompts)
        - **PromptAD paper path**: -> `vision_promptad`
