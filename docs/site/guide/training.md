---
title: 训练
---

# 训练

=== "中文"

    `pyimgano-train` 通过 JSON 配置文件驱动训练流程，支持 recipe、预检、干跑和产物导出。

=== "English"

    `pyimgano-train` drives the training process via JSON config files, with support for recipes, preflight checks, dry runs, and artifact export.

---

## 基本用法

```bash
# 使用配置文件训练
pyimgano-train --config train_config.json

# 指定设备
pyimgano-train --config train_config.json --device cuda:0

# 干跑模式 (验证配置，不实际训练)
pyimgano-train --config train_config.json --dry-run

# 预检验证
pyimgano-train --config train_config.json --preflight
```

---

## 配置文件格式

```json
{
  "recipe": "industrial-adapt",
  "dataset": {
    "root": "/data/my_dataset",
    "train_dir": "train/normal",
    "test_dir": "test",
    "resize": [256, 256]
  },
  "model": {
    "name": "vision_patchcore",
    "feature_extractor": "resnet18",
    "k": 5
  },
  "output": {
    "dir": "./training_output",
    "save_checkpoint": true,
    "export_deploy_bundle": true
  }
}
```

=== "中文"

    配置文件包含四个核心段：

    | 段 | 描述 |
    |----|------|
    | `recipe` | 训练 recipe 名称，定义标准化的训练流程 |
    | `dataset` | 数据集路径、分割方式、预处理参数 |
    | `model` | 模型名称及超参数 |
    | `output` | 输出目录、checkpoint 保存、bundle 导出 |

=== "English"

    The config file has four core sections:

    | Section | Description |
    |---------|-------------|
    | `recipe` | Training recipe name, defining a standardized training process |
    | `dataset` | Dataset paths, split strategy, preprocessing parameters |
    | `model` | Model name and hyperparameters |
    | `output` | Output directory, checkpoint saving, bundle export |

---

## Recipe

=== "中文"

    Recipe 是预定义的训练流程模板，封装了数据加载、预处理、训练循环和输出格式的最佳实践。

=== "English"

    Recipes are predefined training workflow templates that encapsulate best practices for data loading, preprocessing, training loops, and output formats.

```bash
# 列出所有可用 recipe
pyimgano-train --list-recipes

# 使用 industrial-adapt recipe
pyimgano-train --config train_config.json --recipe industrial-adapt
```

!!! note "industrial-adapt"

    `industrial-adapt` 是推荐的工业场景 recipe，包含自适应预处理、训练质量检查和标准化输出。

---

## 训练产物

```bash
training_output/
├── report.json            # 训练报告 (指标、配置、耗时)
├── per_image.jsonl         # 每张图像的训练指标
├── checkpoints/
│   └── model.ckpt         # 模型权重
├── infer_config.json       # 可复用推理配置
└── deploy_bundle/          # 部署包 (可选)
    ├── model.onnx
    └── metadata.json
```

=== "中文"

    | 产物 | 描述 |
    |------|------|
    | `report.json` | 训练摘要：模型指标、配置快照、训练耗时 |
    | `per_image.jsonl` | 每张图像的详细分数，支持逐样本分析 |
    | `checkpoints/` | 模型权重文件，用于恢复或推理 |
    | `infer_config.json` | 推理配置，可直接用于 `pyimgano-infer` |
    | `deploy_bundle/` | 可选的部署包，包含导出模型和元数据 |

=== "English"

    | Artifact | Description |
    |----------|-------------|
    | `report.json` | Training summary: model metrics, config snapshot, training time |
    | `per_image.jsonl` | Per-image detailed scores for sample-level analysis |
    | `checkpoints/` | Model weight files for recovery or inference |
    | `infer_config.json` | Inference config, directly usable by `pyimgano-infer` |
    | `deploy_bundle/` | Optional deployment bundle with exported model and metadata |

---

## 验证与导出

### 干跑与预检

```bash
# 干跑: 解析配置、加载数据元信息，但不实际训练
pyimgano-train --config train_config.json --dry-run

# 预检: 验证数据路径、模型兼容性、依赖完整性
pyimgano-train --config train_config.json --preflight
```

!!! tip "CI 中使用 --preflight"

    在 CI 管道中使用 `--preflight` 快速验证训练配置，无需实际训练即可发现配置错误。

### 导出推理配置

```bash
# 训练完成后导出推理配置
pyimgano-train --config train_config.json --export-infer-config

# 直接导出部署包
pyimgano-train --config train_config.json --export-deploy-bundle
```

=== "中文"

    - `--export-infer-config` — 生成 `infer_config.json`，可直接传给 `pyimgano-infer`
    - `--export-deploy-bundle` — 生成完整部署包，包含模型导出和元数据

=== "English"

    - `--export-infer-config` — Generates `infer_config.json`, directly usable by `pyimgano-infer`
    - `--export-deploy-bundle` — Generates a full deployment bundle with exported model and metadata

---

## 下一步

- [推理](inference.md) — 使用训练产物进行批量推理
- [校准](calibration.md) — 阈值校准与分数标准化
- [基准测试](benchmarking.md) — 系统评测与模型对比
