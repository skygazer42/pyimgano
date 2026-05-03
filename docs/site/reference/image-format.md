# 图像格式参考

=== "中文"

    pyimgano 通过 `ImageFormat` 枚举和 `normalize_numpy_image()` 函数统一处理各种图像像素格式，确保模型接收正确的输入。

=== "English"

    pyimgano uses the `ImageFormat` enum and `normalize_numpy_image()` function to unify various image pixel formats, ensuring models receive correct input.

---

## ImageFormat 枚举

| 枚举值 | dtype | 形状 | 值域 | 说明 |
|--------|-------|------|------|------|
| `RGB_U8_HWC` | `uint8` | `(H, W, 3)` | `[0, 255]` | RGB 通道，HWC 排列 |
| `BGR_U8_HWC` | `uint8` | `(H, W, 3)` | `[0, 255]` | BGR 通道（OpenCV 默认） |
| `GRAY_U8_HW` | `uint8` | `(H, W)` | `[0, 255]` | 8 位灰度 |
| `GRAY_U16_HW` | `uint16` | `(H, W)` | `[0, 65535]` | 16 位灰度（工业相机） |
| `RGB_F32_CHW` | `float32` | `(3, H, W)` | `[0.0, 1.0]` | RGB 浮点，CHW 排列（深度学习张量） |

---

## normalize_numpy_image()

```python
from pyimgano.inputs import ImageFormat, normalize_numpy_image

normalized = normalize_numpy_image(
    image,
    source_format=ImageFormat.BGR_U8_HWC,
    u16_max=4095,  # 可选：12 位传感器最大值
)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `image` | `numpy.ndarray` | 输入图像数组 |
| `source_format` | `ImageFormat` | 输入图像的像素格式 |
| `u16_max` | `int` | 16 位图像的实际最大值（默认 `65535`） |

=== "中文"

    该函数将任意支持格式的图像转换为模型期望的内部格式，自动处理通道顺序、数值类型和归一化。

=== "English"

    This function converts images from any supported format to the model's expected internal format, handling channel order, dtype, and normalization automatically.

---

## 生产集成模式

### OpenCV 帧

```python
import cv2
from pyimgano.inputs import ImageFormat, normalize_numpy_image

cap = cv2.VideoCapture(0)
ret, frame = cap.read()  # BGR_U8_HWC

normalized = normalize_numpy_image(frame, source_format=ImageFormat.BGR_U8_HWC)
result = infer(model, normalized)
```

### 工业相机（12 位/16 位传感器）

```python
import numpy as np
from pyimgano.inputs import ImageFormat, normalize_numpy_image

# 12 位线扫相机输出，存储为 uint16
raw_frame = camera.grab()  # shape: (H, W), dtype: uint16, range: [0, 4095]

normalized = normalize_numpy_image(
    raw_frame,
    source_format=ImageFormat.GRAY_U16_HW,
    u16_max=4095,  # 12 位传感器
)
```

!!! warning "u16_max 参数"

    工业相机常用 10 位（1023）、12 位（4095）或 14 位（16383）传感器。必须根据实际传感器位深设置 `u16_max`，否则图像归一化结果将不正确。

### 深度学习张量

```python
import torch
from pyimgano.inputs import ImageFormat, normalize_numpy_image

# 已在 PyTorch 中预处理的张量
tensor = preprocess(image)  # shape: (3, H, W), dtype: float32, range: [0, 1]
numpy_image = tensor.numpy()

normalized = normalize_numpy_image(numpy_image, source_format=ImageFormat.RGB_F32_CHW)
```

---

## 最佳实践

| 场景 | 推荐格式 | 注意事项 |
|------|----------|----------|
| OpenCV 采集 | `BGR_U8_HWC` | OpenCV 默认 BGR，无需手动转换 |
| PIL/Pillow 加载 | `RGB_U8_HWC` | Pillow 默认 RGB |
| 工业线扫相机 | `GRAY_U16_HW` | 设置正确的 `u16_max` |
| 面阵工业相机（灰度） | `GRAY_U8_HW` | 8 位灰度最常见 |
| PyTorch DataLoader | `RGB_F32_CHW` | 注意值域为 `[0, 1]` |
| 预训练模型输入 | 查看 `infer_config.json` | 以配置中的 `input_format` 为准 |

!!! tip "格式不确定时"

    使用 `pyimgano-doctor` 检查模型期望的输入格式，或查看训练输出的 `infer_config.json` 中的 `input_format` 字段。
