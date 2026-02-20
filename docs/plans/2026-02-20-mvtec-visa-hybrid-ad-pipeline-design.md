# MVTec + VisA Hybrid AD Pipeline (Default-Strong) — Design

**Date:** 2026-02-20  
**Status:** Approved  
**Owner:** @codex (with user approval)

## Background

`pyimgano` aims to be an “enterprise-grade” visual anomaly detection toolkit. Today it includes a large set of
algorithms and utilities, but the **default workflow for MVTec AD + VisA** is not yet engineered as a
repeatable, high-quality, “default is strong” pipeline, especially when we care about **both**:

- **Image-level** anomaly detection (classification)
- **Pixel-level** anomaly localization (heatmaps/masks)

At the same time, the open-source ecosystem (notably Intel’s `anomalib`) has already codified many best
practices: consistent metrics, post-processing, and solid model implementations (PatchCore/EfficientAD/FastFlow).

This design introduces a **Hybrid** approach:

- Keep `pyimgano` as the **unified API + dataset/eval/benchmark/postprocess** layer
- Add **optional** backends (`pyimgano[anomalib]`, `pyimgano[faiss]`) to reach strong defaults quickly
- Preserve a light “core” install

## Goals

1. **Default-strong pipeline** for **MVTec AD + VisA** that supports:
   - one-class training (train on normal-only) as default
   - optional few-shot improvements (calibration and/or training) as an opt-in path
2. **Consistent API semantics** across models:
   - `decision_function(X)` returns continuous anomaly scores
   - `predict(X)` returns 0/1 labels (using a threshold strategy)
   - pixel-level outputs use a consistent method name (e.g. `predict_anomaly_map(X)` or `get_anomaly_map(path)`)
3. **Consistent evaluation** across datasets and models:
   - image-level metrics: AUROC, AP, F1-max/threshold selection
   - pixel-level metrics: Pixel AUROC, Pixel AP, PRO/AUPRO (FPR-limited integration)
4. **Post-processing** for heatmaps to improve localization stability:
   - smoothing, normalization
   - morphological ops
   - connected-component filtering
5. **Optional dependencies**:
   - core stays light: `pip install pyimgano`
   - “strong defaults” extras: `pip install pyimgano[anomalib,faiss]`

## Non-Goals

- Rewriting every deep model implementation to match `anomalib` internals.
- Shipping pretrained weights.
- Implementing a full training framework; we will rely on existing PyTorch/PyOD patterns and optional backends.

## Options Considered

### A) Hybrid (Recommended)

`pyimgano` provides:
- datasets (MVTec/VisA)
- metrics/eval
- post-processing
- a stable detector interface

Model implementations can be:
- “native” `pyimgano` implementations where stable
- optional “backend” wrappers that call `anomalib` (when installed)

**Pros:** fast to reach high quality, core remains light.  
**Cons:** requires clear boundaries and output normalization to keep results comparable.

### B) anomalib-first

`pyimgano` becomes mostly a wrapper around `anomalib`.

**Pros:** fastest.  
**Cons:** weakens `pyimgano` identity; core install becomes less useful.

### C) Native-only

Implement everything ourselves.

**Pros:** full control and minimal deps.  
**Cons:** high risk and time cost; easy to miss “small but important” benchmark/eval details.

## Proposed Architecture

### 1) Dataset Layer

- Keep: `pyimgano.utils.datasets.MVTecDataset` (improve correctness/consistency)
- Add: `VisADataset` with a clear contract:
  - returns images, labels, and masks (if present)
  - exposes a `DatasetInfo`
- Ensure `load_dataset()` supports `visa` in addition to `mvtec`, `btad`, and `custom`.

### 2) Detector Interface

Standardize the following methods:

- **`fit(train_images, ...)`**: trains on normal-only by default
- **`decision_function(test_images)`**: returns continuous anomaly score per image
- **`predict(test_images)`**: returns 0/1 labels, derived from scores + threshold strategy
- **Pixel-level (optional)**:
  - `predict_anomaly_map(test_images)` for batch usage (recommended)
  - keep `get_anomaly_map(path)` as a single-image convenience for existing models

We will support both “native” detectors and optional backend detectors, but the **outputs must follow
the same score/heatmap conventions** so they can be benchmarked together.

### 3) Evaluation

Split evaluation into:

- image-level evaluation: `(y_true, y_scores) -> metrics`
- pixel-level evaluation: `(pixel_labels, pixel_scores) -> metrics`

Default pixel-level metrics:
- Pixel AUROC
- Pixel AP
- PRO/AUPRO with a default integration limit (commonly `0.3`)

### 4) Post-Processing

Provide a configurable post-processing pipeline for pixel anomaly maps:

- normalization (min-max, robust scaling)
- smoothing (Gaussian blur)
- morphological open/close
- remove small connected components

These parameters can be tuned in few-shot mode.

### 5) Optional Backends

Add `extras`:

- `pyimgano[anomalib]`: enables wrappers around anomalib models/metrics
- `pyimgano[faiss]`: enables fast kNN for PatchCore-like memory-bank scoring

Core `pyimgano` does not require these packages.

## Rollout / Work Plan

Implementation plan will be tracked in:

- `docs/plans/2026-02-20-mvtec-visa-hybrid-ad-pipeline.md`

## References

- anomalib: https://github.com/openvinotoolkit/anomalib
- Spot-the-Diff / VisA: https://github.com/amazon-science/spot-diff

