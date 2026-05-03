# Python API 参考

=== "中文"

    本页列出 pyimgano 核心模块的公开 API，按功能模块组织。

=== "English"

    This page lists the public API of pyimgano's core modules, organized by functional area.

---

## pyimgano.models

=== "中文"

    模型创建与注册表。pyimgano 内置 120+ 种异常检测模型。

=== "English"

    Model creation and registry. pyimgano ships with 120+ anomaly detection models.

```python
from pyimgano.models import create_model, list_models

# 列出所有可用模型
models = list_models()

# 创建模型实例
model = create_model("patchcore", device="cuda")
```

| 函数/对象 | 说明 |
|-----------|------|
| `create_model(name, **kwargs)` | 按名称创建模型实例 |
| `list_models()` | 返回所有已注册模型名称列表 |
| `registry` | 模型注册表对象，支持查询元数据 |

---

## pyimgano.inference

=== "中文"

    推理与阈值校准接口。

=== "English"

    Inference and threshold calibration interface.

```python
from pyimgano.inference import infer, infer_iter, calibrate_threshold

# 单次推理
result = infer(model, image)

# 迭代推理（批量）
for result in infer_iter(model, image_dir):
    print(result.score, result.label)

# 阈值校准
threshold = calibrate_threshold(model, calibration_images)
```

| 函数 | 说明 |
|------|------|
| `infer(model, image, **kwargs)` | 对单张图像执行推理，返回检测结果 |
| `infer_iter(model, source, **kwargs)` | 迭代推理，逐张返回结果 |
| `calibrate_threshold(model, images, **kwargs)` | 基于正常样本校准异常分数阈值 |

---

## pyimgano.inputs

=== "中文"

    图像输入格式处理。详细说明参见 [图像格式参考](image-format.md)。

=== "English"

    Image input format handling. See [Image Format Reference](image-format.md) for details.

```python
from pyimgano.inputs import ImageFormat, normalize_numpy_image

# 将 OpenCV BGR 帧转换为标准格式
normalized = normalize_numpy_image(frame, source_format=ImageFormat.BGR_U8_HWC)
```

| 符号 | 说明 |
|------|------|
| `ImageFormat` | 图像格式枚举（RGB_U8_HWC, BGR_U8_HWC 等） |
| `normalize_numpy_image(image, source_format, **kwargs)` | 将任意格式图像归一化为模型期望格式 |

---

## pyimgano.evaluation

=== "中文"

    检测评估与指标计算。

=== "English"

    Detection evaluation and metric computation.

```python
from pyimgano.evaluation import evaluate_detector, compute_auroc, compute_average_precision

# 完整评估
metrics = evaluate_detector(predictions, ground_truth)

# 单项指标
auroc = compute_auroc(scores, labels)
ap = compute_average_precision(scores, labels)
```

| 函数 | 说明 |
|------|------|
| `evaluate_detector(predictions, ground_truth)` | 综合评估，返回多项指标 |
| `compute_auroc(scores, labels)` | 计算 AUROC |
| `compute_average_precision(scores, labels)` | 计算平均精度 (AP) |

---

## pyimgano.synthesis

=== "中文"

    合成异常生成器，用于数据增强和测试。

=== "English"

    Synthetic anomaly generator for data augmentation and testing.

```python
from pyimgano.synthesis import AnomalySynthesizer, SynthSpec, get_preset_names

# 查看可用预设
presets = get_preset_names()

# 创建合成器
synth = AnomalySynthesizer(spec=SynthSpec(preset="industrial"))

# 生成合成异常
anomaly_image, mask = synth.generate(normal_image)
```

| 符号 | 说明 |
|------|------|
| `AnomalySynthesizer` | 合成异常生成器类 |
| `SynthSpec` | 合成规格配置 |
| `get_preset_names()` | 返回可用预设名称列表 |

---

## pyimgano.features

=== "中文"

    特征提取器创建与管理。

=== "English"

    Feature extractor creation and management.

```python
from pyimgano.features import create_feature_extractor, list_feature_extractors

# 列出可用特征提取器
extractors = list_feature_extractors()

# 创建特征提取器
extractor = create_feature_extractor("wide_resnet50", layers=["layer2", "layer3"])
```

| 函数 | 说明 |
|------|------|
| `create_feature_extractor(name, **kwargs)` | 创建特征提取器实例 |
| `list_feature_extractors()` | 返回可用特征提取器列表 |

---

## pyimgano.preprocessing

=== "中文"

    图像预处理流水线。

=== "English"

    Image preprocessing pipeline.

```python
from pyimgano.preprocessing import ImageEnhancer, PreprocessingPipeline

# 创建预处理流水线
pipeline = PreprocessingPipeline([
    ImageEnhancer(brightness=1.2, contrast=1.1),
])

# 应用预处理
processed = pipeline(image)
```

| 类 | 说明 |
|----|------|
| `ImageEnhancer` | 图像增强器（亮度、对比度等） |
| `PreprocessingPipeline` | 预处理流水线，组合多个处理步骤 |

---

## pyimgano.defects

=== "中文"

    缺陷区域提取 API。

=== "English"

    Defect region extraction API.

```python
from pyimgano.defects import extract_defects

defects = extract_defects(anomaly_map, threshold=0.5)
for defect in defects:
    print(defect.bbox, defect.area, defect.score)
```

---

## pyimgano.postprocess

=== "中文"

    异常图后处理。

=== "English"

    Anomaly map post-processing.

```python
from pyimgano.postprocess import AnomalyMapPostprocess

postprocessor = AnomalyMapPostprocess(smooth_sigma=4.0, normalize=True)
processed_map = postprocessor(raw_anomaly_map)
```

| 类 | 说明 |
|----|------|
| `AnomalyMapPostprocess` | 异常图后处理器（平滑、归一化等） |

---

## pyimgano.plugins

=== "中文"

    插件系统，用于加载第三方扩展模型和功能。

=== "English"

    Plugin system for loading third-party extension models and features.

```python
from pyimgano.plugins import load_plugins

# 加载所有已注册插件
load_plugins()
```

| 函数 | 说明 |
|------|------|
| `load_plugins()` | 扫描并加载所有已注册的 pyimgano 插件 |
