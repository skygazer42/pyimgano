---
title: 特征提取器
---

# 特征提取器

=== "中文"

    pyimgano 的特征提取子系统（`pyimgano.features`）提供注册表 API 和运行时协议验证。
    提取器可用于 `vision_embedding_core`、`vision_feature_pipeline` 等模型的特征前端。

=== "English"

    The feature extractor subsystem (`pyimgano.features`) provides a registry API and
    runtime protocol validation. Extractors serve as feature frontends for models like
    `vision_embedding_core` and `vision_feature_pipeline`.

---

## 内置提取器 / Built-in Extractors

| 名称 / Name | 标签 / Tags | 说明 / Description |
|---|---|---|
| `identity` | `embeddings` | 直通输出，用于调试 |
| `hog` | `texture` | 方向梯度直方图 |
| `lbp` | `texture` | 局部二值模式 |
| `gabor_bank` | `texture` | 多尺度 Gabor 滤波组 |
| `color_hist` | `color` | 颜色直方图特征 |
| `edge_stats` | `edges` | 边缘统计特征 |
| `fft_lowfreq` | `frequency` | FFT 低频分量 |
| `patch_stats` | `statistics` | 块统计特征 |
| `multi` | `pipeline` | 多提取器拼接（concat） |
| `pca_projector` | `pipeline` | PCA 降维（fit/transform） |
| `standard_scaler` | `pipeline` | 标准化（fit/transform） |
| `normalize` | `pipeline` | 嵌入归一化 / power transform |
| `torchvision_backbone` | `embeddings` | Torchvision 模型嵌入（全局池化，默认 `pretrained=False`） |
| `torchvision_backbone_gem` | `embeddings` | Torchvision 卷积特征 + GeM 池化（紧凑，强基线） |
| `torchvision_multilayer` | `embeddings` | Torchvision 多层嵌入（拼接） |
| `torchvision_vit_tokens` | `embeddings` | Torchvision ViT token 嵌入 |
| `torchscript_embed` | `embeddings` | TorchScript 自定义嵌入（需检查点路径，离线安全） |
| `openclip_embed` | `embeddings` | OpenCLIP 嵌入（需 `pyimgano[clip]`） |

---

## CLI 发现 / CLI Discovery

```bash
# 列出所有可用特征提取器
pyimgano-benchmark --list-feature-extractors

# JSON 格式输出
pyimgano-benchmark --list-feature-extractors --json

# 按标签筛选
pyimgano-benchmark --list-feature-extractors --feature-tags texture

# 查看单个提取器详情
pyimgano-benchmark --feature-info hog
pyimgano-benchmark --feature-info hog --json
```

---

## Python API

### 创建提取器 / Create Extractor

```python
from pyimgano.features import create_feature_extractor, resolve_feature_extractor

# 按名称创建
extractor = create_feature_extractor("hog")

# 带参数创建
hog = create_feature_extractor("hog", resize_hw=(128, 128))

# 从 spec 字典解析
spec = {"name": "hog", "kwargs": {"pixels_per_cell": (16, 16)}}
extractor = resolve_feature_extractor(spec)
```

### 使用提取器 / Use Extractor

```python
import numpy as np

image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
features = extractor.extract(image)  # -> np.ndarray

# 批量提取
X = extractor.extract([img0, img1])  # -> (2, n_features)
```

### 在模型中使用 / Use in Models

```python
from pyimgano.models import create_model

det = create_model(
    "vision_ecod",
    contamination=0.1,
    feature_extractor={"name": "hog", "kwargs": {"resize_hw": [64, 64]}},
)
```

---

## JSON Spec 格式 / JSON Spec Format

=== "中文"

    在配置文件中，提取器通过 JSON spec 格式指定。支持字符串简写和完整字典两种形式。

=== "English"

    In config files, extractors are specified via JSON spec format. Both string shorthand
    and full dict forms are supported.

```json
// 字符串简写
"feature_extractor": "hog"

// 完整字典
"feature_extractor": {
  "name": "hog",
  "kwargs": {
    "pixels_per_cell": [16, 16],
    "orientations": 9
  }
}

// 组合提取器（multi）
"feature_extractor": {
  "name": "multi",
  "kwargs": {
    "extractors": [
      "hog",
      {"name": "lbp", "kwargs": {"radius": 2}},
      "edge_stats"
    ]
  }
}
```

---

## 流水线提取器 / Pipeline Extractors

=== "中文"

    流水线提取器允许在提取后进行降维或标准化。

=== "English"

    Pipeline extractors allow dimensionality reduction or standardization after extraction.

### multi — 多提取器拼接

```json
{
  "name": "multi",
  "kwargs": {
    "extractors": ["hog", "lbp", "color_hist"]
  }
}
```

### pca_projector — PCA 降维

```json
{
  "name": "pca_projector",
  "kwargs": {
    "base_extractor": "hog",
    "n_components": 50
  }
}
```

### standard_scaler — 标准化

```json
{
  "name": "standard_scaler",
  "kwargs": {
    "base_extractor": "hog"
  }
}
```

### normalize — 嵌入归一化

```json
{
  "name": "normalize",
  "kwargs": {
    "base_extractor": "torchvision_backbone"
  }
}
```

=== "中文"

    `normalize` 对嵌入向量做 power/L2 归一化，特别适合深度提取器输出。

=== "English"

    `normalize` applies power/L2 normalization to embeddings, particularly useful for deep extractor outputs.

---

## 自定义提取器 / Custom Extractor

=== "中文"

    自定义提取器需实现 `extract(inputs) -> np.ndarray` 接口。可选实现 `fit(inputs, y=None) -> self`。

=== "English"

    Custom extractors must implement `extract(inputs) -> np.ndarray`. Optionally implement `fit(inputs, y=None) -> self`.

```python
import numpy as np

class MyExtractor:
    """Custom feature extractor contract."""

    name = "my_extractor"

    def extract(self, inputs) -> np.ndarray:
        """
        Parameters
        ----------
        inputs : np.ndarray or list[np.ndarray]
            Input image(s), HWC uint8.

        Returns
        -------
        np.ndarray
            Feature vector(s), shape (N, D) or (D,).
        """
        # your implementation
        ...
```

=== "中文"

    注册自定义提取器后可通过 `create_feature_extractor("my_extractor")` 使用。
    注册方式参见[插件系统](plugins.md)。

=== "English"

    After registering a custom extractor, use it via `create_feature_extractor("my_extractor")`.
    See [Plugin System](plugins.md) for registration.

!!! note "torchvision_backbone 输入格式"

    === "中文"

        `torchvision_backbone` 接受多种输入格式：`PIL.Image.Image`、`torch.Tensor`、
        HWC numpy 数组 `(H,W,C)` 以及 CHW numpy 数组 `(C,H,W)`。

    === "English"

        `torchvision_backbone` accepts multiple input formats: `PIL.Image.Image`,
        `torch.Tensor`, HWC numpy `(H,W,C)`, and CHW numpy `(C,H,W)`.
