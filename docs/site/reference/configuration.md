# 配置参考

=== "中文"

    pyimgano 使用 JSON 格式的配置文件驱动训练和推理流程。本页详述各配置节的结构与字段。

=== "English"

    pyimgano uses JSON configuration files to drive training and inference workflows. This page details the structure and fields of each configuration section.

---

## 训练配置

=== "中文"

    训练配置是 `pyimgano-train --config` 接受的主配置文件，控制数据集、模型、训练参数和输出行为。

=== "English"

    The training config is the main configuration file accepted by `pyimgano-train --config`, controlling dataset, model, training parameters, and output behavior.

```json
{
  "recipe": "industrial_adapt_audited",
  "dataset": {
    "name": "mvtec_ad",
    "root": "/data/datasets/mvtec",
    "category": "bottle",
    "resize": [256, 256],
    "input_mode": "RGB_U8_HWC"
  },
  "model": {
    "name": "patchcore",
    "device": "cuda",
    "contamination": 0.01
  },
  "output": {
    "save_run": true,
    "output_dir": "runs/bottle_patchcore"
  },
  "training": {
    "epochs": 1,
    "batch_size": 32,
    "num_workers": 4
  },
  "adaptation": {
    "enabled": true,
    "strategy": "auto"
  },
  "defects": {
    "enabled": false,
    "preset": "default"
  }
}
```

### 字段说明

#### `recipe`

| 字段 | 类型 | 说明 |
|------|------|------|
| `recipe` | `string` | 配方名称。使用 `--list-recipes` 查看可用值 |

#### `dataset`

| 字段 | 类型 | 说明 |
|------|------|------|
| `name` | `string` | 数据集名称 |
| `root` | `string` | 数据集根目录路径 |
| `category` | `string` | 数据集类别（如 MVTec AD 中的 `bottle`） |
| `resize` | `[int, int]` | 输入尺寸 `[H, W]` |
| `input_mode` | `string` | 输入图像格式，对应 `ImageFormat` 枚举值 |

#### `model`

| 字段 | 类型 | 说明 |
|------|------|------|
| `name` | `string` | 模型名称 |
| `device` | `string` | 设备（`cuda`, `cpu`, `auto`） |
| `contamination` | `float` | 训练集中异常样本的预估比例 |

#### `output`

| 字段 | 类型 | 说明 |
|------|------|------|
| `save_run` | `bool` | 是否保存运行记录 |
| `output_dir` | `string` | 输出目录路径 |

#### `training`

| 字段 | 类型 | 说明 |
|------|------|------|
| `epochs` | `int` | 训练轮数 |
| `batch_size` | `int` | 批大小 |
| `num_workers` | `int` | 数据加载线程数 |

#### `adaptation`

| 字段 | 类型 | 说明 |
|------|------|------|
| `enabled` | `bool` | 是否启用自适应策略 |
| `strategy` | `string` | 自适应策略（`auto`, `manual`） |

#### `defects`

| 字段 | 类型 | 说明 |
|------|------|------|
| `enabled` | `bool` | 是否在训练后提取缺陷区域 |
| `preset` | `string` | 缺陷提取预设名称 |

---

## 推理配置 (infer_config.json)

=== "中文"

    推理配置在训练完成后自动生成（使用 `--export-infer-config` 或 `--export-deploy-bundle`），包含推理时所需的全部参数。

=== "English"

    The inference config is auto-generated after training (via `--export-infer-config` or `--export-deploy-bundle`), containing all parameters needed for inference.

```json
{
  "model": {
    "name": "patchcore",
    "version": "0.8.0",
    "input_format": "RGB_U8_HWC",
    "input_size": [256, 256]
  },
  "threshold": {
    "image_level": 0.65,
    "pixel_level": 0.45,
    "source": "calibrated"
  },
  "preprocessing": {
    "resize": [256, 256],
    "normalize": true
  },
  "postprocessing": {
    "smooth_sigma": 4.0,
    "normalize_map": true
  }
}
```

!!! tip "验证"

    使用 `pyimgano-validate-infer-config` 验证推理配置的完整性和正确性。

---

## 预设系统

=== "中文"

    预设（Preset）是一组预定义的参数组合，用于简化常见场景的配置。模型预设和缺陷预设可在 CLI 和配置文件中使用。

=== "English"

    Presets are predefined parameter combinations that simplify configuration for common scenarios. Model presets and defect presets can be used in CLI and config files.

```bash
# 使用模型预设推理
pyimgano-infer --model patchcore --model-preset high_recall --input images/

# 使用缺陷提取预设
pyimgano-infer --model patchcore --train-dir runs/model --input images/ \
    --defects --defects-preset strict
```

---

## 套件/扫描配置

=== "中文"

    套件（Suite）和扫描（Sweep）配置用于批量基准测试，定义模型、数据集和参数组合。

=== "English"

    Suite and Sweep configurations are used for batch benchmarking, defining model, dataset, and parameter combinations.

```json
{
  "suite_name": "industrial_core",
  "datasets": ["mvtec_ad", "btad"],
  "models": ["patchcore", "padim", "stfpm"],
  "common": {
    "resize": [256, 256],
    "device": "cuda"
  },
  "sweep": {
    "model.contamination": [0.0, 0.01, 0.05]
  }
}
```

| 字段 | 说明 |
|------|------|
| `suite_name` | 套件名称 |
| `datasets` | 数据集列表 |
| `models` | 模型列表 |
| `common` | 所有组合共享的参数 |
| `sweep` | 要扫描的参数及其取值列表 |
