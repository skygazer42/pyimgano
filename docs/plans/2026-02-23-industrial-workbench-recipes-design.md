# PyImgAno Industrial Workbench (CLI-first + Recipes) — Design

**Date:** 2026-02-23  
**Status:** Approved (user: algorithm engineer focus, CLI-first)  

## Context

`pyimgano` is an open-source toolkit for **industrial visual anomaly detection**. As of `v0.5.8`, it
already provides:

- A large **model registry** (`pyimgano.models.create_model/list_models/model_info`)
- Industrial benchmarking and artifacts (`pyimgano-benchmark`, `runs/<ts>.../report.json`, JSONL)
- Numpy-first inference helpers and high-resolution **tiling** inference
- Optional backends (e.g. anomalib checkpoint wrappers, OpenCLIP)

The next step is to make the library feel like a dependable “workbench” for
**algorithm engineers** working on real industrial projects:

- Business adaptation (augmentations, postprocess, tiling, thresholding, feature extractors)
- Lightweight finetuning (“micro-finetune”) where it adds real value
- A unified, reproducible CLI surface that turns experiments into comparable artifacts

This is intentionally **not** a replacement for PyTorch/TensorFlow, and **not**
a pure sklearn/PyOD clone. It is a pragmatic industrial R&D layer on top of
those ecosystems.

## Positioning (Open Source Identity)

**Recommended positioning:**

> **Industrial Visual Anomaly Detection Workbench** — a recipe-driven toolkit that standardizes the
> industrial loop: dataset alignment → adaptation/finetune → evaluation → reproducible artifacts →
> inference.

**How we relate to existing ecosystems:**

- **PyTorch / TensorFlow**: training/runtime frameworks we rely on (we do not re-implement them).
- **scikit-learn / PyOD**: estimator ergonomics and classical baselines we integrate with.
- **anomalib**: a strong model training stack; `pyimgano` focuses on cross-model *workflows* and
  industrial evaluation + artifacts, with optional checkpoint wrappers.

## Goals

### Must-have

1) **CLI-first workflows** for algorithm engineers.
2) **Recipe-driven adaptation** as the “default path” (augmentation, postprocess, tiling, thresholds).
3) **Micro-finetune** support for a small, high-value subset of deep models.
4) **Reproducibility**:
   - stable run directories
   - run metadata (`config.json`, `environment.json`)
   - per-image JSONL
   - seed support and provenance
5) **Extensibility**:
   - register/extend recipes without editing core CLI logic
   - consistent signatures and JSON-friendly introspection outputs

### Non-goals

- Re-creating a general-purpose deep learning framework (DDP/XLA/etc.).
- Supporting full training/fine-tuning for every model in the registry immediately.
- Packaging model weights inside the wheel (weights stay external and cached on disk).

## Target User Journey (Algorithm Engineer)

### The “daily loop”

1) Pick dataset and category scope (`mvtec`, `visa`, or custom).
2) Choose a baseline recipe preset (fast/balanced/accurate).
3) Run adaptation-first benchmark:
   - augmentations (industrial drift + defect synthesis)
   - tiling for high-res images
   - postprocess for maps
   - threshold calibration
4) Compare runs by artifacts, not screenshots.
5) If needed, enable micro-finetune:
   - small number of epochs / low-rank adapters / last-layer tuning (per recipe)
6) Export the best run’s config + checkpoint to inference.

## Architecture

### Layer 1 — Core Contracts (Hard)

Maintain a minimal detector contract for evaluation:

- `fit(X, y=None) -> self`
- `decision_function(X) -> np.ndarray (N,)`
- `predict(X) -> np.ndarray (N,)` (`{0,1}`)
- optional pixel maps (`get_anomaly_map` / `predict_anomaly_map`)

This is enforced by contract tests (`tests/contracts/...`).

### Layer 2 — Registry + Introspection (Stable API)

The registry remains the single creation/discovery surface:

- `pyimgano.models.create_model`
- `pyimgano.models.model_info` (JSON-friendly payload, computed capabilities)

Recipes will build detectors via the registry and must keep kwargs compatibility
stable via introspection-based filtering.

### Layer 3 — Workbench (Recipes + Training + Artifacts)

Add a “workbench layer” that standardizes industrial workflows:

#### A) Recipes

Introduce `pyimgano.recipes` (or `pyimgano.workbench`) with:

- A `Recipe` protocol (inputs: dataset config, model config, adaptation config; outputs: run artifacts).
- A recipe registry (`register_recipe("industrial-adapt", ...)`) similar to model registry.

Recipes are responsible for:

- adaptation steps (augmentation/postprocess/tiling/threshold calibration)
- optionally invoking micro-finetune for supported models
- producing canonical artifacts

#### B) Training (Micro-finetune)

Introduce a minimal training runner for a small set of supported deep models.
Keep scope narrow:

- deterministic seeding
- checkpoint save/load
- metric logging
- optional early stopping

The training runner should *not* try to be a universal trainer; it is recipe-driven.

#### C) Artifact Schema

Standardize both benchmark and training runs under a compatible layout:

```
runs/<ts>_<dataset>_<entrypoint>/
  report.json
  config.json
  environment.json
  categories/<cat>/report.json
  categories/<cat>/per_image.jsonl
  checkpoints/...
  artifacts/...
```

## CLI Surface (CLI-first)

Keep existing CLIs:

- `pyimgano-benchmark`
- `pyimgano-robust-benchmark`
- `pyimgano-infer`

Add a new CLI:

- `pyimgano-train` — recipe-driven adaptation + micro-finetune + artifact writing.

Key design principles:

- `--config` is the primary interface; flags override config when needed.
- discovery commands (`--list-*`) remain available and lightweight.
- error messages must include actionable install hints.

## Weights / Cache Strategy

Do not ship weights in the wheel.

Use a consistent local cache policy:

- default cache directory under `~/.cache/pyimgano` (or platform equivalent)
- allow overrides via env vars (e.g. `PYIMGANO_CACHE_DIR`, reuse `TORCH_HOME` when appropriate)
- store metadata (model name, version, checksum) alongside cached weights

## Delivery Strategy (Milestones)

Work is executed as **40 tasks** grouped into milestones:

- Milestone 1: tasks 1–10
- Milestone 2: tasks 11–20
- Milestone 3: tasks 21–30
- Milestone 4: tasks 31–40

**Release rule (user requirement):**

- after each milestone:
  - squash to one commit on `main`
  - bump version + update changelog
  - tag and push

## Acceptance Criteria

1) Algorithm engineers can run a full “adaptation-first” workflow from a single CLI entrypoint.
2) Runs are reproducible: artifacts include `config.json` + `environment.json` + stable schema version.
3) A small, supported set of models can micro-finetune end-to-end and then run `pyimgano-infer`.
4) All tests pass on CI (Linux/macOS/Windows, Python 3.9–3.12).

