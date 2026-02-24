# pyimgano（中文）

面向工业质检场景的 **图像异常检测 / 缺陷检测工具包**（支持图像级 + 像素级）。

`pyimgano` 更偏“工程落地”：

- 统一的 **模型注册表**（120+ 模型入口，含可选后端/别名）
- 可复现实验与产物（`pyimgano-train` 工作台 + 报告 + per-image JSONL）
- 部署友好的推理输出（`pyimgano-infer` 输出 JSONL，可选导出缺陷 mask + 连通域区域）
- 工业 IO（numpy-first、显式图像格式、支持高分辨率 tiling）

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

### 训练（workbench）→ 导出 `infer_config.json`

从模板配置开始（需要把数据路径改成你自己的）：

```bash
pyimgano-train \
  --config examples/configs/industrial_adapt_defects_roi.json \
  --export-infer-config
```

默认会生成 `runs/` 下的一个 run 目录，其中包含：

- `artifacts/infer_config.json`（模型 + 阈值 + 后处理 + defects 配置）
- `report.json` / `per_image.jsonl`（可审计的运行产物）

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

## 可选依赖

部分模型/后端需要额外安装：

```bash
pip install "pyimgano[diffusion]"   # diffusion 系列方法
pip install "pyimgano[clip]"        # OpenCLIP 后端
pip install "pyimgano[faiss]"       # kNN 加速（memory bank 类方法）
pip install "pyimgano[anomalib]"    # anomalib checkpoint 包装（偏推理）
pip install "pyimgano[backends]"    # clip + faiss + anomalib
pip install "pyimgano[all]"         # 全量（dev/docs/backends/diffusion/viz）
```

## 文档入口

- `docs/QUICKSTART.md`
- `docs/WORKBENCH.md`
- `docs/CLI_REFERENCE.md`
- `docs/INDUSTRIAL_INFERENCE.md`（numpy-first IO、tiling、defects 导出）
- `docs/MODEL_INDEX.md`（模型索引）

