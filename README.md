# pyimgano

Production-oriented **visual anomaly detection** (image-level + pixel-level) for industrial inspection.

`pyimgano` focuses on the practical parts that matter in production:

- **Unified model registry** (120+ registered model entry points; native implementations + optional backends + aliases)
- **Reproducible CLI runs** (workbench + reports + per-image JSONL)
- **Deploy-friendly inference** (`pyimgano-infer` → JSONL; optional defect masks + connected-component regions)
- **Industrial IO** (numpy-first, explicit image formats, high-resolution tiling)
- **Benchmarking & metrics** (image-level + pixel-level; AUROC/AP/AUPRO/SegF1, etc.)
- **Data & preprocessing** (dataset helpers + preprocessing/augmentation utilities)

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/pyimgano.svg)](https://pypi.org/project/pyimgano/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pyimgano.svg)](https://pypi.org/project/pyimgano/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/skygazer42/pyimgano/actions/workflows/ci.yml/badge.svg)](https://github.com/skygazer42/pyimgano/actions/workflows/ci.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **Translations:** [中文](README_cn.md) · [日本語](README_ja.md) · [한국어](README_ko.md)

## Contents

- [Installation](#installation)
- [Quickstart (CLI)](#quickstart-cli)
- [Quickstart (Python)](#quickstart-python)
- [Models & discovery](#models--discovery)
- [Industrial outputs (defects)](#industrial-outputs-defects)
- [Optional dependencies](#optional-dependencies)
- [Weights & cache policy](#weights--cache-policy)
- [Docs](#docs)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

## Installation

```bash
pip install pyimgano
```

> Note: `pip install pyimgano` works after publishing to PyPI. Until then, install from source:
>
> ```bash
> git clone https://github.com/skygazer42/pyimgano.git
> cd pyimgano
> pip install -e ".[dev]"
> ```
>
> For the release workflow, see `docs/PUBLISHING.md`.

## Quickstart (CLI)

### Train (workbench) → export `infer_config.json`

Start from the provided template and edit dataset paths:

```bash
pyimgano-train \
  --config examples/configs/industrial_adapt_defects_roi.json \
  --export-infer-config
```

This writes a run directory (under `runs/` by default) containing:

- `artifacts/infer_config.json` (model + threshold + postprocess + defects config)
- `report.json` and `per_image.jsonl` (auditable run artifacts)

### Inference → JSONL (+ optional defect masks/regions)

```bash
pyimgano-infer \
  --infer-config /path/to/run_dir/artifacts/infer_config.json \
  --input /path/to/images \
  --defects \
  --save-masks /tmp/pyimgano_masks \
  --save-jsonl /tmp/pyimgano_results.jsonl
```

Guides:
- `docs/WORKBENCH.md` (train/export flow)
- `docs/CLI_REFERENCE.md` (all flags + JSONL schema)
- `docs/INDUSTRIAL_INFERENCE.md` (tiling + defects + ROI notes)

### One-off inference (no workbench)

For quick experiments you can run `pyimgano-infer` directly from a registered model name:

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

Notes:
- Pass extra constructor kwargs via `--model-kwargs '{"backbone":"wide_resnet50","coreset_sampling_ratio":0.1}'`.
- `--defects` requires anomaly maps. If you don’t pass a fixed `--pixel-threshold`, provide `--train-dir` so the default `normal_pixel_quantile` strategy can calibrate one from normal pixels.
- High-resolution tiling works best with detectors tagged `numpy,pixel_map`:
  - add `--tile-size 512 --tile-stride 384` (see `docs/INDUSTRIAL_INFERENCE.md`).

## Quickstart (Python)

### Create a detector from the registry

```python
from pyimgano.models import create_model

detector = create_model(
    "vision_patchcore",
    device="cuda",         # or "cpu"
    contamination=0.1,     # used for threshold defaults when applicable
)

detector.fit(train_paths)                 # normal/reference images
scores = detector.decision_function(test_paths)
```

### Classical baseline (CPU-friendly, no pixel maps)

Classical detectors usually expect a `feature_extractor` that turns images into 2D features:

```python
from pyimgano.models import create_model
from pyimgano.utils import ImagePreprocessor

extractor = ImagePreprocessor(resize=(224, 224), output_tensor=False)
detector = create_model("vision_ecod", feature_extractor=extractor, contamination=0.1, n_jobs=-1)
detector.fit(train_paths)
scores = detector.decision_function(test_paths)
```

### Numpy-first industrial inference

If you already have decoded frames/images in memory, prefer the explicit IO helpers:

```python
import numpy as np

from pyimgano.inference import calibrate_threshold, infer
from pyimgano.inputs import ImageFormat

train_frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(8)]
detector.fit(train_frames)
calibrate_threshold(detector, train_frames, input_format=ImageFormat.RGB_U8_HWC, quantile=0.995)

results = infer(
    detector,
    train_frames[:2],
    input_format=ImageFormat.RGB_U8_HWC,
    include_maps=True,
)
print(results[0].score, results[0].label)
```

Guide: `docs/INDUSTRIAL_INFERENCE.md`

## Models & discovery

`pyimgano` ships **120+ registered model entry points** (native implementations + optional backend wrappers + aliases).
Some models require optional extras (see below).

Discover models from the CLI:

```bash
pyimgano-benchmark --list-models
pyimgano-benchmark --list-models --tags vision,deep
pyimgano-benchmark --model-info vision_patchcore --json
```

### Recommended baselines (practical)

If you’re doing industrial inspection with anomaly maps / defect localization, start here:

| Goal | Start with | Notes |
|------|------------|-------|
| Strong pixel localization baseline | `vision_patchcore` | `numpy,pixel_map`; great default for MVTec/VisA-style data |
| Robustness to “noisy normal” | `vision_softpatch` | `numpy,pixel_map`; filters outlier patches in the memory bank |
| Lightweight pixel baseline | `vision_padim` / `vision_spade` | `numpy,pixel_map`; simpler + often easier to tune |
| Few-shot / small normal set | `vision_anomalydino` | `numpy,pixel_map`; may download DINOv2 weights on first run |
| CPU-only / precomputed features | `vision_ecod` / `vision_copod` | fast, parameter-free; typically score-only (no pixel maps) |
| You already train in anomalib | `vision_*_anomalib` / `vision_anomalib_checkpoint` | requires `pyimgano[anomalib]`; wraps trained checkpoints for evaluation/inference |

For a longer discussion, see `docs/ALGORITHM_SELECTION_GUIDE.md`.

### Tags & capabilities

Every registry entry has `tags` + `metadata`. Useful tags include:

- `classical` vs `deep`
- `pixel_map` → model exposes anomaly maps (`--include-maps`, `--defects`, pixel metrics)
- `numpy` → model can score **RGB uint8 numpy images** (needed for tiling + robustness corruptions)

To inspect a model’s constructor signature and supported kwargs:

```bash
pyimgano-benchmark --model-info vision_patchcore
pyimgano-benchmark --model-info vision_patchcore --json
```

References:
- `docs/MODEL_INDEX.md` (auto-generated model index)
- `docs/ALGORITHM_SELECTION_GUIDE.md` (how to pick a baseline)
- `docs/SOTA_ALGORITHMS.md` and `docs/DEEP_LEARNING_MODELS.md` (deeper dives)

## Industrial outputs (defects)

When `--defects` is enabled, `pyimgano-infer` derives defect structures from the anomaly map:

- binary defect mask (optional artifact output)
- connected-component regions (area / bbox / per-region score)
- optional ROI gating and binary morphology (open/close/fill holes)

This is intended for downstream inspection systems that need more than a single anomaly score.

Docs:
- `docs/CLI_REFERENCE.md` (JSONL schema + examples)
- `docs/INDUSTRIAL_INFERENCE.md` (defects + ROI configuration)

## Optional dependencies

`pyimgano` is usable with the default dependencies, but some models/backends require extras:

```bash
pip install "pyimgano[diffusion]"   # diffusion-based methods
pip install "pyimgano[clip]"        # OpenCLIP backends
pip install "pyimgano[faiss]"       # faster kNN for memory-bank methods
pip install "pyimgano[anomalib]"    # anomalib checkpoint wrappers (inference-first)
pip install "pyimgano[backends]"    # clip + faiss + anomalib
pip install "pyimgano[all]"         # everything (dev/docs/backends/diffusion/viz)
```

## Weights & cache policy

- `pyimgano` does **not** ship model weights inside the wheel.
- When models download weights (torchvision / OpenCLIP / HuggingFace), weights are cached on disk by upstream libraries.
- Common cache env vars you may want to set on servers/containers:
  - `TORCH_HOME`
  - `HF_HOME` / `TRANSFORMERS_CACHE`
  - `XDG_CACHE_HOME`

## Docs

Start here:
- `docs/QUICKSTART.md`
- `docs/WORKBENCH.md`
- `docs/CLI_REFERENCE.md`

Production/industrial:
- `docs/INDUSTRIAL_INFERENCE.md` (numpy-first IO, tiling, postprocess, defects export)
- `docs/ROBUSTNESS_BENCHMARK.md` (drift corruptions + evaluation)
- `docs/MANIFEST_DATASET.md` (JSONL manifest datasets)

Reference:
- `docs/MODEL_INDEX.md`
- `docs/PREPROCESSING.md`
- `docs/EVALUATION_AND_BENCHMARK.md`
- `docs/PUBLISHING.md` (PyPI release flow)

## Contributing

See `CONTRIBUTING.md`.

## License

MIT. See `LICENSE`.

## Citation

```bibtex
@software{pyimgano2026,
  author = {PyImgAno Contributors},
  title = {pyimgano: Production-oriented Visual Anomaly Detection},
  year = {2026},
  url = {https://github.com/skygazer42/pyimgano}
}
```
