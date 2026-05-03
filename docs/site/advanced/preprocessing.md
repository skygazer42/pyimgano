---
title: 预处理
---

# 预处理

=== "中文"

    pyimgano 提供三层预处理能力：基础增强、高级增强和流水线编排。
    此外还包含数据增广工具和光照归一化方法。

=== "English"

    pyimgano provides three levels of preprocessing: basic enhancement, advanced enhancement,
    and pipeline orchestration. It also includes data augmentation tools and illumination
    normalization methods.

---

## ImageEnhancer — 基础图像增强

=== "中文"

    `ImageEnhancer` 提供 80+ 种经典图像处理操作，按类别分组如下。

=== "English"

    `ImageEnhancer` provides 80+ classical image processing operations grouped as follows.

### 边缘检测 / Edge Detection (7)

| 方法 / Method | 说明 / Description |
|---|---|
| `canny` | Canny 边缘检测 |
| `sobel` | Sobel 梯度 |
| `scharr` | Scharr 梯度（更精确） |
| `laplacian` | Laplacian 二阶导数 |
| `prewitt` | Prewitt 梯度 |
| `roberts` | Roberts 交叉梯度 |
| `gabor_edge` | Gabor 滤波边缘 |

### 形态学操作 / Morphological Operations (7)

| 方法 / Method | 说明 / Description |
|---|---|
| `erode` | 腐蚀 |
| `dilate` | 膨胀 |
| `opening` | 开运算 |
| `closing` | 闭运算 |
| `gradient` | 形态学梯度 |
| `tophat` | 顶帽变换 |
| `blackhat` | 黑帽变换 |

### 滤波器 / Filters (4)

| 方法 / Method | 说明 / Description |
|---|---|
| `gaussian_blur` | 高斯模糊 |
| `median_blur` | 中值滤波 |
| `bilateral` | 双边滤波 |
| `sharpen` | 锐化 |

### 归一化 / Normalization (4)

| 方法 / Method | 说明 / Description |
|---|---|
| `minmax` | Min-Max 归一化 |
| `zscore` | Z-score 标准化 |
| `clahe` | CLAHE 直方图均衡 |
| `hist_equalize` | 全局直方图均衡 |

```python
from pyimgano.preprocessing import ImageEnhancer

enhancer = ImageEnhancer()
result = enhancer.apply(image, method="clahe")
result = enhancer.apply(image, method="canny", kwargs={"threshold1": 50, "threshold2": 150})
```

---

## AdvancedImageEnhancer — 高级图像增强

=== "中文"

    `AdvancedImageEnhancer` 提供 25+ 种高级特征空间变换，涵盖频域操作、纹理分析、色彩空间转换、高级去噪和图像分割。

=== "English"

    `AdvancedImageEnhancer` provides 25+ advanced feature-space transforms covering frequency domain operations, texture analysis, color space conversion, advanced denoising, and image segmentation.

| 功能 / Feature | 说明 / Description |
|---|---|
| 色彩空间转换 | HSV, LAB, YCrCb, HLS, LUV 等 |
| Retinex | MSRCR-lite 光照归一化 |
| LBP | 局部二值模式纹理 |
| HOG | 方向梯度直方图 |
| FFT | 频域特征提取与滤波 |
| Gabor | 多尺度 Gabor 滤波 |
| GLCM | 灰度共生矩阵纹理统计 |
| NLM | Non-local means 去噪 |
| Watershed | 分水岭分割 |

```python
from pyimgano.preprocessing import AdvancedImageEnhancer

adv = AdvancedImageEnhancer()
result = adv.apply(image, method="retinex")
result = adv.apply(image, method="lbp")
```

### 频域操作 / Frequency Domain

```python
# FFT 分析
magnitude, phase = adv.apply_fft(image)
reconstructed = adv.apply_ifft(magnitude, phase)

# 频域滤波
lowpass = adv.frequency_filter(image, filter_type='lowpass', cutoff_frequency=30)
highpass = adv.frequency_filter(image, filter_type='highpass', cutoff_frequency=30)
bandpass = adv.frequency_filter(image, filter_type='bandpass', cutoff_frequency=20)
```

### 纹理分析 / Texture Analysis

```python
# Gabor 滤波: 检测定向纹理 (织物、木纹)
gabor = adv.gabor_filter(image, frequency=0.1, theta=np.pi/4)

# LBP: 快速纹理分类，光照不变
lbp = adv.compute_lbp(image, n_points=8, radius=1.0, method='uniform')

# GLCM: 统计纹理分析
glcm_features = adv.compute_glcm(image)
# 返回: contrast, dissimilarity, homogeneity, energy, correlation
```

### 高级去噪 / Advanced Denoising

```python
# Non-local means: 最高质量去噪，较慢
denoised = adv.nlm_denoise(noisy_image, h=10, template_window_size=7)

# 各向异性扩散: 保边平滑
smoothed = adv.anisotropic_diffusion(noisy_image, niter=10, kappa=50)
```

### 图像分割 / Segmentation

```python
# 自动阈值
otsu = adv.threshold(image, method='otsu')
adaptive = adv.threshold(image, method='adaptive_gaussian', block_size=11, c=2)

# 分水岭分割
segmented = adv.watershed(image)
```

### 图像金字塔 / Image Pyramids

```python
# 高斯金字塔 (多尺度表示)
gaussian_pyr = adv.build_gaussian_pyramid(image, levels=4)

# 拉普拉斯金字塔 (带通分解)
laplacian_pyr = adv.build_laplacian_pyramid(image, levels=4)
```

---

## PreprocessingPipeline — 流水线编排

=== "中文"

    `PreprocessingPipeline` 将多个预处理步骤串联为顺序执行的流水线，支持方法链和复用。

=== "English"

    `PreprocessingPipeline` chains multiple preprocessing steps into a sequential pipeline with method chaining and reuse support.

```python
from pyimgano.preprocessing import PreprocessingPipeline

pipeline = PreprocessingPipeline([
    ("clahe", {}),
    ("gaussian_blur", {"ksize": 3}),
    ("canny", {"threshold1": 50, "threshold2": 150}),
])
result = pipeline(image)
```

=== "中文"

    通过 `PreprocessingMixin`，检测器可以内置预处理流水线，使训练和推理自动应用相同的预处理。

=== "English"

    Through `PreprocessingMixin`, detectors can embed a preprocessing pipeline so that
    training and inference automatically apply the same preprocessing.

### PreprocessingMixin 方法 / Methods

| 方法 / Method | 说明 / Description |
|---|---|
| `setup_preprocessing()` | 初始化预处理配置 |
| `add_preprocessing_step()` | 向流水线添加步骤 |
| `preprocess_image()` | 预处理单张图像 |
| `preprocess_images()` | 预处理多张图像 |
| `preprocess_with_edges()` | 快捷: 边缘检测预处理 |
| `preprocess_with_blur()` | 快捷: 模糊预处理 |
| `preprocess_with_morphology()` | 快捷: 形态学预处理 |
| `clear_preprocessing_pipeline()` | 清空流水线 |
| `get_preprocessing_info()` | 获取配置信息 |

```python
from pyimgano.preprocessing import PreprocessingMixin
from pyimgano.models import ECOD

class ECODWithPreprocessing(PreprocessingMixin, ECOD):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.setup_preprocessing(enable=True, use_pipeline=True)
        self.add_preprocessing_step('gaussian_blur', ksize=(5, 5))
        self.add_preprocessing_step('normalize', method='minmax')

    def fit(self, X, y=None):
        X_processed = self.preprocess_images(X)
        X_flat = [img.flatten() for img in X_processed]
        return super().fit(X_flat, y)
```

---

## 光照与对比度归一化 / Illumination & Contrast Normalization

=== "中文"

    在生产环境中，许多误报由**光照漂移**引起：班次间照明变化、相机曝光/白平衡漂移、镜头暗角或不均匀照明。
    pyimgano 提供 opt-in 的 uint8 保持归一化链。

=== "English"

    In production, many false positives are caused by **illumination drift**: lighting changes between shifts,
    camera exposure/white balance drift, lens vignetting or non-uniform illumination.
    pyimgano provides an opt-in, uint8-preserving normalization chain.

```python
from pyimgano.preprocessing import IlluminationContrastKnobs, apply_illumination_contrast

knobs = IlluminationContrastKnobs(
    white_balance="gray_world",
    homomorphic=True,
    clahe=True,
    gamma=0.9,
    contrast_stretch=False,
)
img_normalized = apply_illumination_contrast(img, knobs=knobs)
```

```python
# 流水线风格
from pyimgano.preprocessing import PreprocessingPipeline

pipeline = PreprocessingPipeline().add_step(
    "illumination_contrast",
    white_balance="gray_world",
    homomorphic=True,
    clahe=True,
    gamma=0.9,
)
img_normalized = pipeline.transform(img)
```

### MSRCR-lite Retinex

=== "中文"

    Multi-Scale Retinex with Color Restoration (MSRCR) 的轻量实现，
    用于消除工业场景中的不均匀光照影响。

=== "English"

    A lightweight implementation of Multi-Scale Retinex with Color Restoration (MSRCR)
    for removing uneven illumination effects in industrial scenarios.

```python
from pyimgano.preprocessing import retinex_illumination_normalization

normalized = retinex_illumination_normalization(image)
```

!!! tip "使用场景 / When to Use"

    === "中文"

        - 拍摄环境光照不均匀（如弧形产品表面反光）
        - 不同批次图像亮度差异大
        - 配合 SSIM 模板检测使用效果显著

    === "English"

        - Uneven illumination in capture environment (e.g., reflections on curved surfaces)
        - Large brightness variation across batches
        - Pairs well with SSIM template detection

---

## 数据增广 / Data Augmentation

=== "中文"

    pyimgano 提供面向异常检测场景的数据增广工具，支持组合和随机策略。

=== "English"

    pyimgano provides data augmentation tools tailored for anomaly detection scenarios,
    supporting composition and random strategies.

### 增广操作 / Augmentation Operations

| 操作 / Op | 说明 / Description |
|---|---|
| `RandomRotate` | 随机旋转 |
| `RandomFlip` | 随机翻转 |
| `ColorJitter` | 颜色抖动 |
| `GaussianNoise` | 高斯噪声 |
| `Compose` | 顺序组合 |
| `OneOf` | 随机选一 |

### 预设流水线 / Preset Pipelines

| 预设 / Preset | 强度 / Intensity | 说明 / Description |
|---|---|---|
| `light` | 轻度 | 基础翻转 + 微旋转 |
| `medium` | 中度 | 轻度 + 颜色抖动 |
| `heavy` | 重度 | 中度 + 噪声 + 模糊 |
| `weather` | 天气模拟 | 雨/雾/光照变化 |
| `anomaly` | 异常增广 | 面向缺陷多样化 |

```python
from pyimgano.preprocessing.augmentation import Compose, RandomRotate, RandomFlip, ColorJitter

augment = Compose([
    RandomFlip(p=0.5),
    RandomRotate(degrees=15, p=0.5),
    ColorJitter(brightness=0.1, contrast=0.1, p=0.3),
])
augmented = augment(image)
```

### 工业预设增广 / Industrial Preset Augmentation

```python
from pyimgano.preprocessing import (
    get_industrial_camera_robust_augmentation,
    get_industrial_surface_defect_synthesis_augmentation,
)

# 相机鲁棒性增广 (噪声、模糊、压缩伪影)
camera_aug = get_industrial_camera_robust_augmentation()
img_cam = camera_aug(img)

# 表面缺陷合成增广
defect_aug = get_industrial_surface_defect_synthesis_augmentation()
img_defect = defect_aug(img)
```

=== "中文"

    低级缺陷合成函数也可直接使用，保持 `(H, W, C)` uint8 契约，可组合到自定义流水线：

=== "English"

    Low-level defect synthesis functions can also be used directly, preserving the `(H, W, C)` uint8 contract and composable into custom pipelines:

```python
from pyimgano.preprocessing.augmentation import add_scratches, add_dust, add_specular_highlight

scratched = add_scratches(image)
dusty = add_dust(image)
highlighted = add_specular_highlight(image)
```

---

## 最佳实践 / Best Practices

### 操作顺序 / Operation Order

=== "中文"

    预处理步骤的顺序至关重要。推荐顺序：

    1. **去噪** — Gaussian、bilateral、median
    2. **增强** — sharpen、CLAHE
    3. **特征提取** — 边缘、形态学
    4. **归一化** — 始终放在最后

=== "English"

    The order of preprocessing steps is critical. Recommended order:

    1. **Denoising** — Gaussian, bilateral, median
    2. **Enhancement** — sharpen, CLAHE
    3. **Feature extraction** — edges, morphology
    4. **Normalization** — always last

```python
# 正确: 先去噪再检测边缘
pipeline.add_step('gaussian_blur', ksize=(5, 5))  # 先去噪
pipeline.add_step('detect_edges', method='canny')  # 再检测边缘

# 错误: 在噪声图像上检测边缘会包含噪声边缘
```

### 场景化预设 / Scenario-Based Presets

```python
# 表面缺陷 (纹理异常)
pipeline = PreprocessingPipeline()
pipeline.add_step('gaussian_blur', ksize=(3, 3))
pipeline.add_step('unsharp_mask', sigma=1.0, amount=1.5)
pipeline.add_step('normalize', method='minmax')

# 结构异常 (形状/边缘)
pipeline = PreprocessingPipeline()
pipeline.add_step('bilateral_filter', d=9, sigma_color=75, sigma_space=75)
pipeline.add_step('detect_edges', method='canny', threshold1=50, threshold2=150)
pipeline.add_step('dilate', kernel_size=(3, 3))
pipeline.add_step('normalize', method='minmax')

# 对比度问题
pipeline = PreprocessingPipeline()
pipeline.add_step('clahe', clip_limit=2.0)
pipeline.add_step('normalize', method='robust')
```

!!! tip "计算成本 / Computational Cost"

    === "中文"

        平衡质量与速度：实时场景用 Gaussian + MinMax，离线分析用 bilateral + CLAHE + unsharp + robust。

    === "English"

        Balance quality and speed: use Gaussian + MinMax for real-time, bilateral + CLAHE + unsharp + robust for offline analysis.
