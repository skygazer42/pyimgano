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

## 已验证 artifact 推理

直接加载单个 artifact：

```bash
pyimgano-infer \
  --artifact ./exports/bottle/native \
  --input ./data/test \
  --save-jsonl ./results.jsonl
```

export root 或 deploy bundle 可能包含多个 runtime。必须将选择缩小到唯一项：

```bash
pyimgano-infer \
  --artifact ./exports \
  --artifact-category bottle \
  --artifact-format onnx \
  --artifact-backend onnxruntime \
  --input ./data/test \
  --save-jsonl ./results.jsonl
```

`--artifact-id sha256:...` 是精确选择器，不能与 category/format/backend selector 组合。
artifact manifest 自己拥有 detector reconstruction、预处理、threshold 与 postprocess policy；
artifact mode 因此拒绝 `--model`、`--train-dir` 等覆盖项。

ONNX provider 选择是显式且有序的：

```bash
pyimgano-infer \
  --artifact ./exports/bottle/onnx \
  --onnx-providers CUDAExecutionProvider,CPUExecutionProvider \
  --onnx-provider-options '{"CUDAExecutionProvider":{"device_id":"0"}}' \
  --onnx-session-options '{"intra_op_num_threads":4}' \
  --input ./data/test \
  --save-jsonl ./results.jsonl
```

raw `.onnx` 不能直接传给 `--artifact`。先按照
[训练产物导出与第三方导入](../deployment/export.md)使用 `pyimgano-artifact import`
和显式语义契约导入。

TorchScript artifact（`single_graph` 与 `composite`）默认不会执行反序列化。确认来源后必须
显式开启 trust：

```bash
pyimgano-infer \
  --artifact ./exports/bottle/torchscript \
  --trust-checkpoint \
  --input ./data/test \
  --save-jsonl ./results.jsonl
```

PyTorch 官方 [`torch.jit.load`](https://docs.pytorch.org/docs/stable/generated/torch.jit.load.html)
安全说明指出恶意模型可能执行任意代码；artifact hash 不替代 provenance review。

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
from pyimgano.models import create_model
from pyimgano.inference import infer, infer_iter

# detector 由调用方创建并拟合
detector = create_model("vision_ecod", contamination=0.1)
detector.fit(train_paths)

# 批量推理（path input 不需要 input_format）
results = infer(detector, test_paths)

# 迭代推理（逐张返回，节省内存）
for path, result in zip(test_paths, infer_iter(detector, test_paths)):
    print(f"{path}: score={result.score:.4f}")
```

Artifact runtime 使用同一 detector-compatible API：

```python
from pyimgano.inference import infer, load_artifact

with load_artifact(
    "./exports",
    category="bottle",
    format="native",
) as detector:
    scores = detector.decision_function(["images/a.png", "images/b.png"])
    results = infer(detector, ["images/a.png", "images/b.png"])
```

TorchScript 的 Python 路径同样必须显式传入
`load_artifact(path, trust_checkpoint=True)`。这既适用于 `ae_resnet_unet` 的
`single_graph`，也适用于 `vision_torchscript_ecod` 的 `composite`。

`ArtifactRuntime` 还提供 `predict()`，并在 output contract 声明 map 时提供
`predict_anomaly_map()`。score-only 且没有 operating threshold 的 artifact 会拒绝
`predict()`，但仍可调用 `decision_function()` 与 `infer()`。

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
from pyimgano.inputs import ImageFormat, normalize_numpy_image

# 常见格式
ImageFormat.BGR_U8_HWC    # OpenCV 默认格式 (H, W, 3) uint8
ImageFormat.RGB_U8_HWC    # PIL 默认格式 (H, W, 3) uint8
ImageFormat.GRAY_U8_HW    # 灰度图 (H, W) uint8
ImageFormat.RGB_F32_CHW   # PyTorch 风格 (3, H, W) float32

# 归一化: 将任意格式转为标准内部格式
image_normalized = normalize_numpy_image(
    raw_image,
    input_format=ImageFormat.BGR_U8_HWC,
)
```

!!! tip "生产环境必备"

    在集成工业相机或第三方图像源时，始终使用 `ImageFormat` + `normalize_numpy_image()` 确保格式一致性。

---

## 下一步

- [校准](calibration.md) — 阈值校准与分数标准化
- [缺陷检测](defects.md) — 独立缺陷检测管线详解
- [Python API](python-api.md) — 核心 API 参考
