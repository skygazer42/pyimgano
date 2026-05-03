---
title: 像素级基线
---

# 像素级基线配方

=== "中文"

    像素级基线模型直接在像素空间进行异常检测，无需训练或仅需少量参考图像。
    它们返回逐像素的异常分数图（anomaly map），适合快速原型验证和参考检测场景。

=== "English"

    Pixel-level baseline models perform anomaly detection directly in pixel space,
    requiring no training or only a few reference images. They return per-pixel anomaly
    score maps, ideal for quick prototyping and reference-based inspection.

---

## 模型选择指南 / Model Selection Guide

| 模型 / Model | 方法 / Method | 需要参考图 | 对齐要求 | 推荐场景 |
|---|---|:---:|:---:|---|
| `ssim_template_map` | SSIM 模板匹配 | 1 张 | 严格对齐 | 金标准模板对比 |
| `ssim_struct_map` | 结构 SSIM（边缘图） | 1 张 | 中等 | 纹理/结构变化检测 |
| `vision_template_ncc_map` | 局部 NCC 相似度 | 1 张 | 中等 | 局部细节匹配 |
| `vision_phase_correlation_map` | 相位相关对齐 + 差分 | 1 张 | 自动平移对齐 | 轻微位移容错 |
| `vision_pixel_gaussian_map` | 逐像素高斯 z-score | N 张 | 严格对齐 | 统计分布建模 |
| `vision_pixel_mad_map` | 鲁棒 MAD z-score | N 张 | 严格对齐 | 抗离群像素干扰 |
| `vision_pixel_mean_absdiff_map` | 均值绝对差 | N 张 | 严格对齐 | 最简单的基线 |

---

## ssim_template_map — SSIM 模板检测

=== "中文"

    基于结构相似性（SSIM）将测试图像与金标准模板逐窗口比较，输出异常分数图。
    适合产品高度一致、对齐精确的场景。

=== "English"

    Compares test images against a golden template using structural similarity (SSIM)
    in a sliding-window fashion, outputting an anomaly score map. Best for products with
    high consistency and precise alignment.

```bash
pyimgano-infer \
  --model ssim_template_map \
  --data ./data/test/ \
  --reference ./data/golden/template.png \
  --defects \
  --output-dir ./results/ssim/
```

```python
from pyimgano.models import create_model

model = create_model("ssim_template_map")
model.set_reference(reference_image)
anomaly_map = model.predict_anomaly_map(test_image)
```

---

## ssim_struct_map — 结构 SSIM 边缘检测

=== "中文"

    先提取边缘图，再执行 SSIM 比较。对光照变化更鲁棒，聚焦结构性差异。

=== "English"

    Extracts edge maps first, then performs SSIM comparison. More robust to illumination
    changes, focusing on structural differences.

```bash
pyimgano-infer \
  --model ssim_struct_map \
  --data ./data/test/ \
  --reference ./data/golden/template.png \
  --defects \
  --output-dir ./results/struct/
```

```python
model = create_model("ssim_struct_map")
model.set_reference(reference_image)
anomaly_map = model.predict_anomaly_map(test_image)
```

---

## vision_template_ncc_map — 局部 NCC 相似度

=== "中文"

    使用归一化互相关（NCC）在局部窗口内计算相似度，对亮度偏移鲁棒。

=== "English"

    Uses Normalized Cross-Correlation (NCC) in local windows, robust to brightness shifts.

```bash
pyimgano-infer \
  --model vision_template_ncc_map \
  --data ./data/test/ \
  --reference ./data/golden/template.png \
  --defects \
  --output-dir ./results/ncc/
```

---

## vision_phase_correlation_map — 相位相关对齐

=== "中文"

    先通过相位相关自动估计平移偏移并对齐，再计算绝对差异图。
    适合拍摄时存在轻微位移的场景。

=== "English"

    Automatically estimates translation offset via phase correlation, aligns images, then
    computes absolute difference maps. Suitable for scenarios with slight camera shift.

```bash
pyimgano-infer \
  --model vision_phase_correlation_map \
  --data ./data/test/ \
  --reference ./data/golden/template.png \
  --defects \
  --output-dir ./results/phase/
```

---

## vision_pixel_gaussian_map — 逐像素高斯 z-score

=== "中文"

    从多张正常图像学习每个像素的均值和标准差，推理时计算 z-score。
    需要较多参考图像（推荐 >= 20 张）。

=== "English"

    Learns per-pixel mean and standard deviation from multiple normal images, then
    computes z-scores at inference time. Requires more reference images (>= 20 recommended).

```bash
# 训练（学习像素统计）
pyimgano-train fit \
  --model vision_pixel_gaussian_map \
  --data ./data/train/normal/

# 推理
pyimgano-infer \
  --model vision_pixel_gaussian_map \
  --data ./data/test/ \
  --defects \
  --output-dir ./results/gaussian/
```

```python
model = create_model("vision_pixel_gaussian_map")
model.fit(normal_images)  # list of np.ndarray
anomaly_map = model.predict_anomaly_map(test_image)
```

---

## vision_pixel_mad_map — 鲁棒 MAD z-score

=== "中文"

    使用中位数绝对偏差（MAD）替代标准差，对离群像素更鲁棒。
    推荐在正常样本中可能存在少量异常噪声时使用。

=== "English"

    Uses Median Absolute Deviation (MAD) instead of standard deviation, more robust to
    outlier pixels. Recommended when normal samples may contain slight anomalous noise.

```bash
pyimgano-train fit \
  --model vision_pixel_mad_map \
  --data ./data/train/normal/

pyimgano-infer \
  --model vision_pixel_mad_map \
  --data ./data/test/ \
  --defects \
  --output-dir ./results/mad/
```

---

## vision_pixel_mean_absdiff_map — 均值绝对差

=== "中文"

    最简单的基线：计算测试图像与正常图像均值之间的绝对差异。
    适合作为 sanity check 的第一步。

=== "English"

    The simplest baseline: computes the absolute difference between the test image and
    the mean of normal images. Suitable as a first-pass sanity check.

```bash
pyimgano-train fit \
  --model vision_pixel_mean_absdiff_map \
  --data ./data/train/normal/

pyimgano-infer \
  --model vision_pixel_mean_absdiff_map \
  --data ./data/test/ \
  --defects \
  --output-dir ./results/absdiff/
```

---

## Python API 示例 / Python API Example

```python
import numpy as np
from pyimgano.models import create_model

# 选择模型
model = create_model("ssim_template_map")

# 设置参考图像（单模板模型）
reference = np.array(...)  # H x W x C, uint8
model.set_reference(reference)

# 预测异常图
test_image = np.array(...)
anomaly_map = model.predict_anomaly_map(test_image)
# anomaly_map: np.ndarray, shape (H, W), float32, higher = more anomalous

# 带阈值的缺陷检测
score = model.predict(test_image)  # image-level anomaly score
```

!!! tip "选择建议 / Selection Tips"

    === "中文"

        - **模板严格对齐** → `ssim_template_map`
        - **光照不稳定** → `ssim_struct_map` 或 `vision_template_ncc_map`
        - **轻微位移** → `vision_phase_correlation_map`
        - **多张参考图像** → `vision_pixel_gaussian_map` 或 `vision_pixel_mad_map`
        - **快速验证** → `vision_pixel_mean_absdiff_map`

    === "English"

        - **Strict template alignment** → `ssim_template_map`
        - **Unstable illumination** → `ssim_struct_map` or `vision_template_ncc_map`
        - **Slight camera shift** → `vision_phase_correlation_map`
        - **Multiple reference images** → `vision_pixel_gaussian_map` or `vision_pixel_mad_map`
        - **Quick sanity check** → `vision_pixel_mean_absdiff_map`
