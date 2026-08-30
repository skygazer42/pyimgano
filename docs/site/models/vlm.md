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
| WinCLIP / WinCLIP+ | `winclip` / `vision_winclip` | `paper-adaptation` | OpenCLIP ViT-B/16+ | 是 | `open_clip` |
| WinCLIP upstream | `vision_winclip_anomalib` | `external-backend` | anomalib | 是 | `anomalib` |
| AnomalyDINO | `vision_anomalydino` | `paper-adaptation` | DINOv2 ViT-S/14 | 是 | `torch` |
| OpenCLIP PatchKNN | `vision_openclip_patchknn` | `not-applicable` | OpenCLIP | 是 | `open_clip` |
| PromptAD | `vision_promptad` | `paper-adaptation` | VV-CLIP ViT-B/16+ | 是 | `open_clip` |

!!! note "少样本说明"
    AnomalyDINO 是无需参数训练的少样本方法，但仍需要至少 1 张正常参考图建立 patch 记忆库。

!!! warning "论文复现状态"
    本地 `vision_winclip` 与 `vision_anomalydino` 均为论文适配实现。
    论文关系以 `model_info(...)["metadata"]["paper_fidelity"]` 为准。
    `vision_promptad` 也是论文适配实现，但其图像级与像素级提示需分别训练。

---

## WinCLIP / WinCLIP+

=== "中文"

    本地实现使用论文的完整组合提示集、ViT-B/16+ 的 2x2/3x3 masked-token
    窗口、窗口与跨尺度调和聚合。设置 `k_shot > 0` 后，WinCLIP+ 还会建立
    patch/小窗口/中窗口三套正常视觉记忆并与语言分数融合。

=== "English"

    The local implementation uses the paper's complete compositional prompts,
    2x2/3x3 masked-token ViT-B/16+ windows, and harmonic overlap/cross-scale
    aggregation. With `k_shot > 0`, WinCLIP+ adds patch, small-window, and
    mid-window normal memories and fuses them with the language score.

### 关键参数

| 参数 | 默认值 | 说明 |
|:---|:---|:---|
| `class_name` | `"object"` | 组合提示中的对象类别名 |
| `openclip_model_name` | `"ViT-B-16-plus-240"` | 论文默认 OpenCLIP 主干 |
| `openclip_pretrained` | `"laion400m_e31"` | LAION-400M 预训练权重 |
| `image_size` | `240` | 论文输入分辨率 |
| `scales` | `(2, 3)` | patch 网格上的小/中窗口尺度，stride 固定为 1 |
| `temperature` | `0.07` | 二分类余弦 softmax 温度 |
| `k_shot` | `0` | 少样本数量（0 = 零样本） |
| `tile_overlap` | `0.2` | 非方形图像切片最小重叠率 |
| `text_prompts` | `None` | 可选自定义正常/异常完整提示列表 |
| `device` | auto | 计算设备 |

### 零样本检测

```python
from pyimgano import create_model

# 零样本 -- 无需训练数据
model = create_model("winclip",
                     class_name="bottle",
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
                     class_name="bottle",
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

    本地实现采用论文的 DINOv2-S/14 patch 特征、余弦近邻、最高 1% 尾部均值、
    参考图旋转、类别条件 PCA 前景掩码，以及 σ=4 的像素图平滑。

=== "English"

    The local implementation follows the paper's DINOv2-S/14 patch features,
    cosine nearest neighbours, top-1% tail mean, reference rotations,
    category-conditioned PCA foreground masks, and sigma-4 map smoothing.

### 关键参数

| 参数 | 默认值 | 说明 |
|:---|:---|:---|
| `pretrained` | `False` | 是否通过 torch.hub 下载 DINOv2 权重 |
| `embedder` | `None` | 自定义补丁嵌入器（离线推荐） |
| `dino_model_name` | `"dinov2_vits14"` | DINOv2 模型变体 |
| `n_neighbors` | `1` | kNN 近邻数 |
| `coreset_sampling_ratio` | `1.0` | coreset 子采样比例 |
| `image_size` | `448` | 论文默认短边分辨率；保持宽高比后裁到 14 的倍数 |
| `aggregation_method` | `"topk_mean"` | 图像分数采用 patch 距离最高 1% 的均值 |
| `aggregation_topk` | `0.01` | 论文尾部比例 |
| `reference_rotations` | 8 个 45° 间隔角度 | 内置 DINOv2 的论文默认参考增强 |
| `class_name` | `None` | MVTec/VisA 类别名；用于论文表中的自动掩码选择 |
| `masking` | auto | 可显式覆盖 PCA 前景掩码开关 |
| `gaussian_sigma` | `4.0` | 上采样异常图的论文平滑参数 |
| `device` | `"cpu"` | 计算设备 |

### 基本用法

```python
from pyimgano import create_model

model = create_model("vision_anomalydino",
                     pretrained=True,
                     class_name="capsule",
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
    image_size=448,
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
# WinCLIP / PromptAD / OpenCLIP baselines
pip install pyimgano[clip]

# AnomalyDINO -- torch + torchvision (DINOv2 via torch.hub)
pip install pyimgano[torch]

# 安装所有 VLM 依赖
pip install pyimgano[clip,torch]
```

---

## 选择建议

!!! tip "何时选择 VLM"

    === "中文"

        - **WinCLIP 零样本论文路径**: -> `winclip` (k_shot=0)
        - **WinCLIP+ 少样本论文路径**: -> `winclip` (k_shot=1/2/4)
        - **少量参考图 (1-10 张) 的 kNN 基线**: -> `vision_anomalydino`
        - **需要像素定位 + 少样本免训练**: -> `vision_anomalydino`
        - **自定义语义描述**: -> `winclip` (自定义 text_prompts)
        - **PromptAD 少样本论文路径**: -> `vision_promptad`

    === "English"

        - **WinCLIP zero-shot paper path**: -> `winclip` (k_shot=0)
        - **WinCLIP+ few-shot paper path**: -> `winclip` (k_shot=1/2/4)
        - **Few-reference kNN baseline (1-10)**: -> `vision_anomalydino`
        - **Pixel localization + training-free few-shot**: -> `vision_anomalydino`
        - **Custom semantic descriptions**: -> `winclip` (custom text_prompts)
        - **PromptAD paper path**: -> `vision_promptad`
