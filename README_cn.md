# pyimgano（中文）

面向工业质检场景的 **图像异常检测 / 缺陷检测工具包**（支持图像级 + 像素级）。

`pyimgano` 更偏“工程落地”：

- 统一的 **模型注册表**（120+ 模型入口，含可选后端/别名）
- 可复现实验与产物（`pyimgano-train` 工作台 + 报告 + per-image JSONL）
- 部署友好的推理输出（`pyimgano-infer` 输出 JSONL，可选导出缺陷 mask + 连通域区域）
- 工业 IO（numpy-first、显式图像格式、支持高分辨率 tiling）
- 基准测试与指标（图像级 + 像素级：AUROC/AP/AUPRO/SegF1 等）
- 数据与预处理（数据集工具 + 预处理/增强能力）

> English README: `README.md`（功能更新通常以英文版本为准）

## 目录

- [安装](#安装)
- [快速上手（CLI）](#快速上手cli)
- [快速上手（Python）](#快速上手python)
- [可选依赖](#可选依赖)
- [文档入口](#文档入口)

## 安装

```bash
pip install pyimgano
```

> 说明：当项目发布到 PyPI 后，`pip install pyimgano` 才可直接安装。
> 在此之前建议从源码安装：
>
> ```bash
> git clone https://github.com/skygazer42/pyimgano.git
> cd pyimgano
> pip install -e ".[dev]"
> ```
>
> 发布流程见：`docs/PUBLISHING.md`。

## 快速上手（CLI）

### 最快离线自检（推荐）

跑一个最小端到端 demo（自动生成 tiny `custom` 数据集 + 跑 suite + 导出表格）：

```bash
pyimgano-demo
```

提示：用 `pyimgano-demo --help` 查看可选参数（suite/sweep/output 等）。

可选：先做一次环境自检，看看哪些 baseline 会因为缺少 extras 被跳过：

```bash
pyimgano-doctor --suite industrial-v4
```

统一发现入口：

```bash
pyim --list
pyim --list models --family patchcore
pyim --list models --year 2021 --type deep-vision
pyim --list models --type flow-based
pyim --list model-presets --family graph
pyim --list years --json
pyim --list types --json
pyim --list preprocessing --deployable-only

pyimgano-infer --list-model-presets --family distillation --json
pyimgano-infer --list-models --year 2021 --type deep-vision
pyimgano-infer --list-models --year 2001 --type one-class-svm

pyimgano-infer \
  --model vision_patchcore \
  --preprocessing-preset illumination-contrast-balanced \
  --input /path/to/images
```

### 训练（workbench）→ 导出 `infer_config.json`

从模板配置开始（需要把数据路径改成你自己的）：

```bash
pyimgano-train \
  --config examples/configs/industrial_adapt_defects_fp40.json \
  --export-infer-config
```

默认会生成 `runs/` 下的一个 run 目录，其中包含：

- `artifacts/infer_config.json`（模型 + 阈值 + 后处理 + defects 配置）
- `report.json` / `per_image.jsonl`（可审计的运行产物）

可选（部署 bundle）：把一个目录整体拷贝到服务器/容器：

```bash
pyimgano-train --config cfg.json --export-deploy-bundle
```

### 推理 → JSONL（可选导出缺陷 mask/regions）

```bash
pyimgano-infer \
  --infer-config /path/to/run_dir/artifacts/infer_config.json \
  --input /path/to/images \
  --defects \
  --save-masks /tmp/pyimgano_masks \
  --save-jsonl /tmp/pyimgano_results.jsonl
```

相关文档：
- `docs/WORKBENCH.md`
- `docs/CLI_REFERENCE.md`
- `docs/INDUSTRIAL_INFERENCE.md`
- `docs/FALSE_POSITIVE_DEBUGGING.md`（overlay + 降误报调参流程）

### 快速推理（不走 workbench）

你也可以直接用注册表里的模型名做一次性推理：

```bash
pyimgano-infer \
  --model vision_patchcore \
  --preset industrial-balanced \
  --device cuda \
  --train-dir /path/to/normal/train_images \
  --calibration-quantile 0.995 \
  --input /path/to/images \
  --include-maps
```

说明：
- 额外构造参数用 `--model-kwargs '{"backbone":"wide_resnet50","coreset_sampling_ratio":0.1}'` 传入。
- `--defects` 依赖 anomaly map。如果不传固定 `--pixel-threshold`，建议提供 `--train-dir`，这样默认的 `normal_pixel_quantile` 策略可以从 normal pixels 校准一个阈值。
- 高分辨率 tiling 更建议用带 `numpy,pixel_map` tag 的模型（见 `docs/INDUSTRIAL_INFERENCE.md`）。

### 算法选型：基线套件（suite）+ 小网格扫参（sweep）

工业算法选型常见需求是：在一个数据类目上快速对比一组 **baseline**，并得到可审计的汇总报告与排行榜表格。
`pyimgano-benchmark` 支持运行内置的 **suite**（基线套件），并可选启用小范围 `sweep`（每个 baseline 的小网格搜索）。

发现 suite / sweep：

```bash
pyimgano-benchmark --list-suites
pyimgano-benchmark --suite-info industrial-v3 --json

pyimgano-benchmark --list-sweeps
pyimgano-benchmark --sweep-info industrial-template-small --json
```

运行 suite（并导出排行榜表格 `leaderboard.*` / `skipped.*`）：

```bash
pyimgano-benchmark \
  --dataset mvtec \
  --root /path/to/mvtec_ad \
  --category bottle \
  --suite industrial-v3 \
  --device cpu \
  --no-pretrained \
  --suite-export both \
  --output-dir /tmp/pyimgano_suite_run
```

可选：启用 sweep（对每个 baseline 扫少量变体，并限制每个 baseline 的变体数）：

```bash
pyimgano-benchmark \
  --dataset mvtec \
  --root /path/to/mvtec_ad \
  --category bottle \
  --suite industrial-v3 \
  --suite-sweep industrial-template-small \
  --suite-sweep-max-variants 1 \
  --suite-export csv \
  --output-dir /tmp/pyimgano_suite_sweep_run
```

说明：
- 部分 suite 条目是 **optional**：如果没有安装对应 extras，会自动跳过，并给出可执行提示，例如
  `pip install 'pyimgano[skimage]'`、`pip install 'pyimgano[torch]'`。
- 详细参数、输出目录结构见 `docs/CLI_REFERENCE.md`。

## 快速上手（Python）

```python
from pyimgano.models import create_model

detector = create_model(
    "vision_patchcore",
    device="cuda",         # 或 "cpu"
    contamination=0.1,
)

detector.fit(train_paths)
scores = detector.decision_function(test_paths)
```

## 模型与选择建议

常见工业场景（需要 anomaly map / 缺陷定位）推荐从这些基线开始：

| 目标 | 推荐模型 | 备注 |
|------|----------|------|
| 强像素定位基线 | `vision_patchcore` | `numpy,pixel_map`；强默认选择 |
| “noisy normal” 更鲁棒 | `vision_softpatch` | `numpy,pixel_map`；对内存库 patch 做鲁棒过滤 |
| 更轻量的像素基线 | `vision_padim` / `vision_spade` | `numpy,pixel_map`；更容易跑通/调参 |
| few-shot / normal 很少 | `vision_anomalydino` | `numpy,pixel_map`；首次可能下载 DINOv2 权重 |
| CPU-only / 预提特征 | `vision_ecod` / `vision_copod` | 速度快、参数少；通常只有 score（无像素图） |
| 已在 anomalib 训练 | `vision_*_anomalib` / `vision_anomalib_checkpoint` | 需要 `pyimgano[anomalib]`；加载 checkpoint 做评估/推理 |

更完整的选择逻辑见：`docs/ALGORITHM_SELECTION_GUIDE.md`。

### 模型发现（CLI）

```bash
pyimgano-benchmark --list-models
pyimgano-benchmark --list-models --tags numpy,pixel_map
pyimgano-benchmark --model-info vision_patchcore --json
```

## 可选依赖

部分模型/后端需要额外安装：

```bash
pip install "pyimgano[torch]"       # torch + torchvision（深度模型、TorchScript export 等）
pip install "pyimgano[onnx]"        # onnx + onnxruntime（ONNX 推理/导出）
pip install "pyimgano[openvino]"    # OpenVINO runtime（部署推理）
pip install "pyimgano[skimage]"     # scikit-image（SSIM/phase-corr 等基线）
pip install "pyimgano[numba]"       # numba 加速基线
pip install "pyimgano[viz]"         # matplotlib/seaborn 可视化
pip install "pyimgano[diffusion]"   # diffusion 系列方法
pip install "pyimgano[clip]"        # OpenCLIP 后端
pip install "pyimgano[faiss]"       # kNN 加速（memory bank 类方法）
pip install "pyimgano[anomalib]"    # anomalib checkpoint 包装（偏推理）
pip install "pyimgano[backends]"    # clip + faiss + anomalib
pip install "pyimgano[all]"         # 全量（dev/docs/viz + torch/onnx/openvino + backends/diffusion 等）
```

也可以参考：`docs/OPTIONAL_DEPENDENCIES.md`（extras 对照表 + 推荐安装组合）。

## 文档入口

- `docs/QUICKSTART.md`
- `docs/WORKBENCH.md`
- `docs/CLI_REFERENCE.md`
- `docs/INDUSTRIAL_INFERENCE.md`（numpy-first IO、tiling、defects 导出）
- `docs/FALSE_POSITIVE_DEBUGGING.md`（误报排查与调参）
- `docs/MODEL_INDEX.md`（模型索引）
