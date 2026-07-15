<p align="center">
  <img src="https://raw.githubusercontent.com/skygazer42/pyimgano/main/assets/readme-banner.png" alt="pyimgano README banner" width="100%"/>
</p>

<p align="center">
  <strong>Production-oriented visual anomaly detection for industrial inspection.</strong><br/>
  <sub>Image-level + pixel-level · 120+ models · Train → deploy in one pipeline</sub>
</p>

<p align="center">
  <a href="https://pypi.org/project/pyimgano/"><img src="https://img.shields.io/pypi/v/pyimgano?style=flat-square&logo=pypi&logoColor=white&label=PyPI" alt="PyPI"/></a>
  <a href="https://pypi.org/project/pyimgano/"><img src="https://img.shields.io/pypi/pyversions/pyimgano?style=flat-square&logo=python&logoColor=white" alt="Python"/></a>
  <a href="https://github.com/skygazer42/pyimgano/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/skygazer42/pyimgano/ci.yml?style=flat-square&logo=githubactions&logoColor=white&label=CI" alt="CI"/></a>
  <a href="https://github.com/skygazer42/pyimgano/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square" alt="License: MIT"/></a>
</p>

<p align="center">
  <a href="#installation">Installation</a> ·
  <a href="#quickstart">Quickstart</a> ·
  <a href="#recommended-models">Models</a> ·
  <a href="#documentation">Documentation</a>
</p>

<p align="center">
  <b>Translations:</b>
  <a href="https://github.com/skygazer42/pyimgano/blob/main/README_cn.md">中文</a> ·
  <a href="https://github.com/skygazer42/pyimgano/blob/main/README_ja.md">日本語</a> ·
  <a href="https://github.com/skygazer42/pyimgano/blob/main/README_ko.md">한국어</a>
</p>

---

## Why pyimgano?

`pyimgano` helps teams move from model selection to deployable industrial image
inspection without stitching together separate research, benchmark, and runtime tools.

- Unified API for classical, deep, and vision-language anomaly detectors
- Image scores, pixel anomaly maps, masks, regions, and ROI-aware post-processing
- Reproducible train, benchmark, inference, and acceptance CLIs
- Deploy bundles with infer configs, manifests, hashes, and operator handoff reports
- Path and NumPy inputs for offline, batch, and production integrations

See the [project comparison](https://github.com/skygazer42/pyimgano/blob/main/docs/COMPARISON.md)
for when to choose `pyimgano`, PyOD, or anomalib.

---

## Installation

```bash
pip install pyimgano
```

Install a task profile only when you need it:

```bash
pip install "pyimgano[deploy]"       # training + ONNX/OpenVINO deployment
pip install "pyimgano[benchmark]"    # benchmark vision backends
pip install "pyimgano[cpu-offline]"  # richer offline CPU baselines
```

Check the environment or ask the CLI for the smallest missing extra:

```bash
pyimgano --help
python -m pyimgano --help
pyimgano-doctor --profile first-run --json
pyimgano-doctor --recommend-extras --for-command train --json
```

See [Optional Dependencies](https://github.com/skygazer42/pyimgano/blob/main/docs/OPTIONAL_DEPENDENCIES.md)
for Torch, OpenCLIP, FAISS, anomalib, visualization, and contributor installs.

---

## Quickstart

### Python API

Train on normal images, then score unseen images:

```python
from pyimgano.models import create_model

detector = create_model("vision_ecod", contamination=0.1)
detector.fit(train_paths)
scores = detector.decision_function(test_paths)
```

Use `predict(...)` for labels or an image model with the `pixel_map` capability when
you also need localization.

### First-run CLI

Run the bounded, CPU-friendly smoke path:

```bash
pyimgano-demo --smoke --emit-next-steps --no-pretrained
```

### Guided Workflow

This is the compact Discover → Benchmark → Train → Export → Infer → Validate → Gate path:

```bash
# Discover
pyim --goal first-run --json

# Benchmark
pyimgano benchmark --list-starter-configs
pyimgano benchmark --starter-config-info official_mvtec_industrial_v4_cpu_offline.json --json

# Train + Export
pyimgano train --list-recipes
pyimgano train --recipe-info industrial-adapt --json
pyimgano train --dry-run --config examples/configs/industrial_adapt_audited.json
pyimgano train --config examples/configs/industrial_adapt_audited.json \
  --export-infer-config --export-deploy-bundle

# Infer
pyimgano-infer --from-run runs/<run_dir> --input /path/to/images --save-jsonl results.jsonl

# Validate + Gate
pyimgano-bundle validate runs/<run_dir>/deploy_bundle --json
pyimgano runs acceptance runs/<run_dir> --require-status audited --check-bundle-hashes --json
```

Successful JSONL records include `decision_summary` and, when applicable,
`postprocess_summary`. Deploy bundles include `infer_config.json`,
`bundle_manifest.json`, and `handoff_report.json`.

---

## Recommended Models

| Goal | Start with | Notes |
|---|---|---|
| Pixel localization | `vision_patchcore` | Strong default for MVTec/VisA-style data |
| Noisy normal training data | `vision_softpatch` | Position-wise LOF denoising + soft-weighted patch memory |
| Lightweight anomaly maps | `vision_padim` | Simpler deep baseline with fewer tuning knobs |
| CPU-only scoring | `vision_ecod` or `vision_copod` | Fast classical starting points |
| Few-shot inspection | `vision_anomalydino` | DINOv2-based; weights may download on first use |

Discover models from the CLI instead of scanning a static list:

```bash
pyim --list models --objective latency --selection-profile cpu-screening --topk 5
pyim --goal pixel-localization --json
```

See the [Algorithm Selection Guide](https://github.com/skygazer42/pyimgano/blob/main/docs/ALGORITHM_SELECTION_GUIDE.md)
and [Model Index](https://github.com/skygazer42/pyimgano/blob/main/docs/MODEL_INDEX.md)
for capabilities, dependencies, and trade-offs.

---

## Documentation

| Goal | Guide |
|---|---|
| Run the first smoke test | [Start Here](https://github.com/skygazer42/pyimgano/blob/main/docs/START_HERE.md) |
| Choose an exact route | [Starter Paths](https://github.com/skygazer42/pyimgano/blob/main/docs/STARTER_PATHS.md) |
| Learn the Python API | [Full Quickstart](https://github.com/skygazer42/pyimgano/blob/main/docs/QUICKSTART.md) |
| Browse runnable examples | [Examples Index](https://github.com/skygazer42/pyimgano/blob/main/examples/README.md) |
| Select a model | [Algorithm Selection Guide](https://github.com/skygazer42/pyimgano/blob/main/docs/ALGORITHM_SELECTION_GUIDE.md) |
| Run reproducible benchmarks | [Benchmark Getting Started](https://github.com/skygazer42/pyimgano/blob/main/docs/BENCHMARK_GETTING_STARTED.md) |
| Train and hand off a deploy bundle | [Industrial Fast-Path](https://github.com/skygazer42/pyimgano/blob/main/docs/INDUSTRIAL_FASTPATH.md) |
| Integrate NumPy, JSONL, maps, and defects | [Industrial Inference](https://github.com/skygazer42/pyimgano/blob/main/docs/INDUSTRIAL_INFERENCE.md) |
| Use `bundle watch`, webhook delivery, or `audit-bundle` | [CLI Reference](https://github.com/skygazer42/pyimgano/blob/main/docs/CLI_REFERENCE.md) |
| Compare runs with `pyimgano-runs` and quality gates | [Run Comparison](https://github.com/skygazer42/pyimgano/blob/main/docs/RUN_COMPARISON.md) |
| Understand architecture boundaries | [Classical Pipelines](https://github.com/skygazer42/pyimgano/blob/main/docs/ARCHITECTURE_CLASSICAL_PIPELINES.md) · [Deep Contracts](https://github.com/skygazer42/pyimgano/blob/main/docs/ARCHITECTURE_DEEP_CONTRACTS.md) |
| Browse every document | [Documentation Index](https://github.com/skygazer42/pyimgano/blob/main/docs/README_DOCS.md) |

Published documentation is also available at
[skygazer42.github.io/pyimgano](https://skygazer42.github.io/pyimgano/).

---

## Contributing

See [CONTRIBUTING.md](https://github.com/skygazer42/pyimgano/blob/main/CONTRIBUTING.md),
[CODE_OF_CONDUCT.md](https://github.com/skygazer42/pyimgano/blob/main/CODE_OF_CONDUCT.md),
and [SECURITY.md](https://github.com/skygazer42/pyimgano/blob/main/SECURITY.md).

## License

[MIT](https://github.com/skygazer42/pyimgano/blob/main/LICENSE)

## Citation

GitHub citation metadata is provided in
[CITATION.cff](https://github.com/skygazer42/pyimgano/blob/main/CITATION.cff).

```bibtex
@software{pyimgano2026,
  author = {PyImgAno Contributors},
  title  = {pyimgano: Production-oriented Visual Anomaly Detection},
  year   = {2026},
  url    = {https://github.com/skygazer42/pyimgano}
}
```

<p align="center">
  <sub>Made with care for industrial inspection teams.</sub><br/>
  <a href="https://github.com/skygazer42/pyimgano">⭐ Star pyimgano on GitHub</a>
</p>
