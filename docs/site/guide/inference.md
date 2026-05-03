---
title: 推理
---

# 推理

=== "中文"

    `pyimgano-infer` 支持批量图像推理，输出 JSONL 格式结果，集成分块处理、缺陷检测与多种图像格式。

=== "English"

    `pyimgano-infer` supports batch image inference with JSONL output, integrated tiling, defect detection, and multiple image formats.

---

## CLI 基本用法

```bash
# 基本推理
pyimgano-infer \
  --model vision_patchcore \
  --train-dir ./data/train/normal \
  --input ./data/test \
  --save-jsonl ./results.jsonl

# 使用模型预设
pyimgano-infer \
  --model-preset industrial-template-ncc-map \
  --train-dir ./data/train/normal \
  --input ./data/test \
  --save-jsonl ./results.jsonl
```

=== "中文"

    | 参数 | 描述 |
    |------|------|
    | `--model` | 模型名称（与 `create_model()` 一致） |
    | `--model-preset` | 使用预定义模型配置 |
    | `--train-dir` | 正常样本训练目录 |
    | `--input` | 测试图像目录或单张图像路径 |
    | `--save-jsonl` | 结果输出路径（JSONL 格式） |

=== "English"

    | Flag | Description |
    |------|-------------|
    | `--model` | Model name (same as `create_model()`) |
    | `--model-preset` | Use a predefined model configuration |
    | `--train-dir` | Normal sample training directory |
    | `--input` | Test image directory or single image path |
    | `--save-jsonl` | Result output path (JSONL format) |

---

## JSONL 输出格式

```json
{"path": "test/img_001.png", "score": 0.87, "prediction": 1, "label": "anomalous"}
{"path": "test/img_002.png", "score": 0.12, "prediction": 0, "label": "normal"}
```

=== "中文"

    每行一条 JSON 记录，包含图像路径、异常分数、预测标签和可读标签。方便与下游管道集成。

=== "English"

    One JSON record per line with image path, anomaly score, prediction label, and human-readable label. Easy to integrate with downstream pipelines.

---

## 分块处理 (Tiling)

```bash
# 大图分块推理
pyimgano-infer \
  --model vision_patchcore \
  --train-dir ./data/train/normal \
  --input ./data/test \
  --tile-size 256 \
  --tile-stride 128 \
  --save-jsonl ./results.jsonl
```

=== "中文"

    对于高分辨率图像（如工业相机 4K+ 输出），分块处理将图像切分为固定大小的 tile 独立推理，再合并结果。

    - `--tile-size` — 每块大小（像素）
    - `--tile-stride` — 滑动步长（小于 tile-size 时产生重叠）

=== "English"

    For high-resolution images (e.g., 4K+ industrial camera output), tiling splits images into fixed-size tiles for independent inference, then merges results.

    - `--tile-size` — Tile size in pixels
    - `--tile-stride` — Sliding stride (overlap when less than tile-size)

!!! warning "分块参数选择"

    stride 应不大于 tile-size。较小的 stride 增加重叠区域，提高边缘精度但增加计算量。推荐 stride = tile-size / 2。

---

## 缺陷检测集成

```bash
# 推理 + 缺陷检测
pyimgano-infer \
  --model vision_patchcore \
  --train-dir ./data/train/normal \
  --input ./data/test \
  --defects \
  --defects-preset industrial-default \
  --save-masks ./output/masks \
  --save-overlays ./output/overlays \
  --save-jsonl ./results.jsonl
```

=== "中文"

    | 参数 | 描述 |
    |------|------|
    | `--defects` | 启用缺陷检测后处理 |
    | `--defects-preset` | 缺陷检测预设 |
    | `--save-masks` | 保存二值掩码 |
    | `--save-overlays` | 保存带标注的叠加图 |

=== "English"

    | Flag | Description |
    |------|-------------|
    | `--defects` | Enable defect detection post-processing |
    | `--defects-preset` | Defect detection preset |
    | `--save-masks` | Save binary masks |
    | `--save-overlays` | Save annotated overlay images |

---

## Python API

```python
from pyimgano.inference import infer, infer_iter

# 批量推理 (返回完整结果列表)
results = infer(
    model="vision_patchcore",
    train_dir="./data/train/normal",
    input_path="./data/test",
)

# 迭代推理 (逐张返回，节省内存)
for result in infer_iter(
    model="vision_patchcore",
    train_dir="./data/train/normal",
    input_path="./data/test",
):
    print(f"{result.path}: score={result.score:.4f}")
```

=== "中文"

    - `infer()` — 一次性返回所有结果，适合小批量
    - `infer_iter()` — 生成器模式逐张返回，适合大批量或流式处理

=== "English"

    - `infer()` — Returns all results at once, suitable for small batches
    - `infer_iter()` — Generator mode yielding one result at a time, suitable for large batches or streaming

---

## 图像格式 (ImageFormat)

=== "中文"

    在生产集成中，图像可能来自不同的采集设备，格式各异。使用 `ImageFormat` 显式声明输入格式，避免隐式转换错误。

=== "English"

    In production integration, images may come from different acquisition devices with varying formats. Use `ImageFormat` to explicitly declare the input format, avoiding implicit conversion errors.

```python
from pyimgano.inference import ImageFormat, normalize_numpy_image

# 常见格式
ImageFormat.BGR_U8_HWC    # OpenCV 默认格式 (H, W, 3) uint8
ImageFormat.RGB_U8_HWC    # PIL 默认格式 (H, W, 3) uint8
ImageFormat.GRAY_U8_HW    # 灰度图 (H, W) uint8
ImageFormat.RGB_F32_CHW   # PyTorch 风格 (3, H, W) float32

# 归一化: 将任意格式转为标准内部格式
image_normalized = normalize_numpy_image(raw_image, format=ImageFormat.BGR_U8_HWC)
```

!!! tip "生产环境必备"

    在集成工业相机或第三方图像源时，始终使用 `ImageFormat` + `normalize_numpy_image()` 确保格式一致性。

---

## 下一步

- [校准](calibration.md) — 阈值校准与分数标准化
- [缺陷检测](defects.md) — 独立缺陷检测管线详解
- [Python API](python-api.md) — 核心 API 参考
