<p align="center">
  <img src="https://raw.githubusercontent.com/skygazer42/pyimgano/main/assets/readme-banner-white-art.png" alt="pyimgano README banner" width="100%"/>
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
  <a href="#deployment">Deployment</a> ·
  <a href="#recommended-models">Models</a>
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
pip install "pyimgano[deploy]"       # artifact creation + ONNX/OpenVINO runtimes
pip install "pyimgano[onnx-runtime]" # ONNX import/inference without Torch
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

# Export a run whose model reports native trained-export support
pyimgano-export --from-run runs/<certified_run_dir> --format native --out ./exports

# Infer from a run or verified artifact
pyimgano-infer --from-run runs/<run_dir> --input /path/to/images --save-jsonl results.jsonl
pyimgano-infer --artifact ./exports \
  --artifact-format native --input /path/to/images --save-jsonl artifact-results.jsonl

# Validate + Gate
pyimgano-bundle validate runs/<run_dir>/deploy_bundle --json
pyimgano runs acceptance runs/<run_dir> --require-status audited --check-bundle-hashes --json
```

## Deployment

The current schema-v1 full-format trained-export target is
`ae_resnet_unet` (native supported; graph formats conditional on a complete
checkpoint and backend dependencies). Existing `vision_patchcore` starters do
not claim trained-export support. Source-locked composite certification is also
available for `vision_onnx_ecod` in ONNX format only and
`vision_torchscript_ecod` in TorchScript format only. Every TorchScript artifact
load requires `--trust-checkpoint` or API `trust_checkpoint=True` after
provenance review.

Successful JSONL records include `decision_summary` and, when applicable,
`postprocess_summary`. Deploy bundles include `infer_config.json`,
`bundle_manifest.json`, and `handoff_report.json`. `pyimgano-export --from-run`
is the canonical post-run path for a verified, relocatable fitted detector.

Raw ONNX files must first be imported with an explicit preprocessing/output
contract; they are not accepted as self-describing artifacts:

```bash
pyimgano-artifact import --format onnx --model model.onnx \
  --contract onnx-contract.json --out imported-artifact
pyimgano-infer --artifact imported-artifact --input /path/to/images \
  --save-jsonl results.jsonl
```

See [Trained Artifacts](https://github.com/skygazer42/pyimgano/blob/main/docs/TRAINED_ARTIFACTS.md)
for format support, manifest contracts, trust boundaries, and the Python API.

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
