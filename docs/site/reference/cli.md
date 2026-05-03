# CLI 参考

=== "中文"

    pyimgano 提供 20+ 个命令行工具，覆盖训练、推理、基准测试、部署和环境管理的完整工作流。

=== "English"

    pyimgano provides 20+ CLI tools covering the complete workflow of training, inference, benchmarking, deployment, and environment management.

---

## pyimgano

=== "中文"

    统一入口命令（umbrella CLI），通过子命令访问各功能模块。

=== "English"

    Unified entry-point (umbrella CLI) providing access to all functional modules via subcommands.

```bash
pyimgano <subcommand> [options]
```

| 子命令 | 说明 |
|--------|------|
| `bundle` | 部署包管理（`validate` / `run` / `watch`） |
| `runs` | 运行记录管理（`quality` / `acceptance`） |
| `weights` | 权重管理（`audit-bundle`） |
| `train` | 配方驱动的工作台运行（umbrella，等价于 `pyimgano-train`） |
| `infer` | 推理（umbrella，等价于 `pyimgano-infer`） |
| `list` | 模型 / 数据集 / 特征提取器发现 |

---

## pyim

=== "中文"

    快捷发现命令，用于快速探索可用模型、数据集和功能。

=== "English"

    Discovery shortcut for quickly exploring available models, datasets, and features.

```bash
pyim models          # 列出可用模型
pyim datasets        # 列出已注册数据集
pyim features        # 列出特征提取器

# 按目标推荐 recipe（v0.9.0+ 携带结构化 train 后续命令）
pyim --goal "industrial-adapt-fast" --json
```

---

## pyimgano-benchmark

=== "中文"

    模型基准测试工具，支持单模型、套件和参数扫描模式。

=== "English"

    Model benchmarking tool supporting single-model, suite, and parameter sweep modes.

```bash
# 单模型基准测试
pyimgano-benchmark --dataset mvtec_ad --model patchcore --pixel --save-run

# 套件模式
pyimgano-benchmark --suite industrial_core --save-run

# 参数扫描
pyimgano-benchmark --suite-sweep sweep_config.json
```

| 参数 | 说明 |
|------|------|
| `--dataset` | 数据集名称 |
| `--model` | 模型名称 |
| `--suite` | 测试套件名称 |
| `--suite-sweep` | 套件参数扫描配置文件 |
| `--pixel` | 启用像素级评估 |
| `--save-run` | 保存运行结果 |
| `--list-models` | 列出所有可用模型 |
| `--model-info <name>` | 查看模型详细信息 |

---

## pyimgano-train

=== "中文"

    模型训练入口，支持配置文件驱动、配方系统和一键部署包导出。

=== "English"

    Model training entry point supporting config-driven workflows, recipe system, and one-step deploy bundle export.

```bash
# 基本训练
pyimgano-train --config my_config.json

# 预检
pyimgano-train --config my_config.json --preflight

# 训练并导出部署包
pyimgano-train --config my_config.json --export-deploy-bundle
```

| 参数 | 说明 |
|------|------|
| `--config` | 训练配置文件路径 |
| `--dry-run` | 试运行（mini-epoch） |
| `--preflight` | 配置预检，不执行训练 |
| `--export-infer-config` | 导出推理配置 |
| `--export-deploy-bundle` | 导出完整部署包 |
| `--list-recipes` | 列出所有可用配方 |
| `--recipe-info <name>` | 查看配方详细信息 |

---

## pyimgano-infer

=== "中文"

    推理工具，对图像执行异常检测并输出结果。

=== "English"

    Inference tool for running anomaly detection on images and producing results.

```bash
# 基本推理
pyimgano-infer --model patchcore --train-dir runs/my_model --input test_images/

# 带缺陷提取和可视化
pyimgano-infer --model patchcore --train-dir runs/my_model --input test_images/ \
    --defects --save-masks --save-overlays --save-jsonl results.jsonl
```

| 参数 | 说明 |
|------|------|
| `--model` | 模型名称 |
| `--model-preset` | 模型预设名称 |
| `--train-dir` | 训练输出目录 |
| `--input` | 输入图像或目录 |
| `--save-jsonl` | 输出 JSONL 结果文件路径 |
| `--defects` | 启用缺陷区域提取 |
| `--defects-preset` | 缺陷提取预设 |
| `--save-masks` | 保存异常掩码 |
| `--save-overlays` | 保存叠加可视化图 |
| `--tile-size` | 分块推理块大小 |
| `--tile-stride` | 分块推理步长 |

---

## pyimgano-defects

=== "中文"

    缺陷区域提取与分析工具。

=== "English"

    Defect region extraction and analysis tool.

```bash
pyimgano-defects --input anomaly_maps/ --output defects/
```

---

## pyimgano-bundle

=== "中文"

    部署包管理命令，包含验证、一次性运行和热文件夹长驻三个子命令。

=== "English"

    Deploy bundle management with `validate`, `run`, and `watch` subcommands.

```bash
# 验证 bundle
pyimgano bundle validate my_bundle/ --json

# 一次性运行
pyimgano bundle run my_bundle/ --image-dir test_images/

# 热文件夹长驻（v0.9.0+）
pyimgano-bundle watch my_bundle/ \
    --watch-dir ./inbox \
    --output-dir ./bundle_watch \
    --poll-seconds 1.0 --settle-seconds 2.0
```

### `bundle watch` 关键参数

| 参数 | 说明 |
|------|------|
| `--watch-dir` | 投递目录（被轮询） |
| `--output-dir` | 聚合产物目录 |
| `--once` | 处理当前 backlog 后退出（CI / 批处理友好） |
| `--poll-seconds` | 轮询周期 |
| `--settle-seconds` | 文件稳定窗口（避免读取写入中的文件） |
| `--webhook-url` | 投递每条结果到下游系统 |
| `--webhook-bearer-token{,-env,-file}` | Bearer token 三种来源 |
| `--webhook-signing-secret{,-env,-file}` | HMAC-SHA256 签名密钥三种来源 |
| `--webhook-header KEY=VALUE` | 自定义请求头（可重复） |
| `--webhook-timeout-seconds` | 单次 POST 超时 |
| `--webhook-retry-min-seconds` | 失败重试最小间隔（指数退避下限） |

详细用法参见 [部署包指南](../deployment/bundle.md)。

---

## pyimgano-demo

=== "中文"

    演示工具，快速运行预置场景或冒烟测试。

=== "English"

    Demo tool for quickly running preset scenarios or smoke tests.

```bash
# 冒烟测试
pyimgano-demo --smoke

# 运行特定场景
pyimgano-demo --scenario industrial_pcb
```

| 参数 | 说明 |
|------|------|
| `--smoke` | 执行最小冒烟测试 |
| `--scenario` | 运行指定演示场景 |

---

## pyimgano-doctor

=== "中文"

    环境诊断工具，检查依赖、加速器和兼容性。

=== "English"

    Environment diagnostics tool for checking dependencies, accelerators, and compatibility.

```bash
# 完整环境报告
pyimgano-doctor

# JSON 格式输出
pyimgano-doctor --json

# 检查并推荐缺失依赖
pyimgano-doctor --recommend-extras
```

| 参数 | 说明 |
|------|------|
| `--profile` | 按工作流 profile 检查兼容性（`first-run` / `deploy-smoke` / `benchmark` / `deploy` / `publish`） |
| `--json` | 以 JSON 格式输出 |
| `--recommend-extras` | 推荐需要安装的 extras |
| `--for-command <name>` | 与 `--recommend-extras` 联用，按命令给出安装建议（如 `train` / `export-onnx`） |
| `--for-model <name>` | 与 `--recommend-extras` 联用，按模型给出安装建议 |
| `--accelerators` | 检查可用加速器（torch CUDA/MPS、onnxruntime providers、openvino devices） |
| `--require-extras <name>` | 断言指定 extras 已安装（可重复，用作 CI 门控；缺失时退出码 1） |
| `--suite <name>` | 按 benchmark suite 报告会因缺失 extras 被跳过的基线 |
| `--run-dir <path>` | 评估某个 run 的 readiness / acceptance 就绪度 |
| `--deploy-bundle <path>` | 评估某个 deploy bundle 的就绪度 |

---

## pyimgano-synthesize

=== "中文"

    合成异常样本生成工具，用于数据增强和模型测试。

=== "English"

    Synthetic anomaly sample generation tool for data augmentation and model testing.

```bash
pyimgano-synthesize --input normal_images/ --output synthetic/ --preset industrial
```

---

## pyimgano-evaluate

=== "中文"

    评估工具，计算检测指标（AUROC、AP、F1 等）。

=== "English"

    Evaluation tool for computing detection metrics (AUROC, AP, F1, etc.).

```bash
pyimgano-evaluate --predictions results.jsonl --ground-truth labels/
```

---

## pyimgano-manifest

=== "中文"

    数据集清单管理工具。

=== "English"

    Dataset manifest management tool.

```bash
pyimgano-manifest create --dataset-dir ./my_dataset --output manifest.json
```

---

## pyimgano-datasets

=== "中文"

    数据集管理与信息查询。

=== "English"

    Dataset management and information query.

```bash
pyimgano-datasets list
pyimgano-datasets info mvtec_ad
```

---

## pyimgano-weights

=== "中文"

    权重文件管理，包括下载、缓存和审计。

=== "English"

    Weight file management including download, caching, and auditing.

```bash
pyimgano-weights list
pyimgano-weights download <model_name>
pyimgano-weights audit-bundle my_bundle/
```

---

## pyimgano-runs

=== "中文"

    训练运行记录管理，包括质量检查和验收检查。

=== "English"

    Training run management including quality checks and acceptance checks.

```bash
pyimgano runs list
pyimgano runs quality <run_dir>
pyimgano runs acceptance <run_dir>
```

---

## pyimgano-robust-benchmark

=== "中文"

    鲁棒性基准测试，评估模型在扰动条件下的表现。

=== "English"

    Robustness benchmarking for evaluating model performance under perturbation conditions.

```bash
pyimgano-robust-benchmark --model patchcore --dataset mvtec_ad --perturbations noise,blur,contrast
```

---

## pyimgano-features

=== "中文"

    特征提取器管理与信息查询。

=== "English"

    Feature extractor management and information query.

```bash
pyimgano-features list
pyimgano-features info wide_resnet50
```

---

## pyimgano-export-onnx

=== "中文"

    将模型导出为 ONNX 格式。

=== "English"

    Export model to ONNX format.

```bash
pyimgano-export-onnx --train-dir runs/my_model --output embed.onnx
```

详细用法参见 [模型导出](../deployment/export.md)。

---

## pyimgano-export-torchscript

=== "中文"

    将模型导出为 TorchScript 格式。

=== "English"

    Export model to TorchScript format.

```bash
pyimgano-export-torchscript --train-dir runs/my_model --output embed.ts
```

详细用法参见 [模型导出](../deployment/export.md)。

---

## pyimgano-validate-infer-config

=== "中文"

    验证推理配置文件的完整性和正确性。

=== "English"

    Validate inference configuration file completeness and correctness.

```bash
pyimgano-validate-infer-config infer_config.json
pyimgano-validate-infer-config my_bundle/infer_config.json --strict
```
