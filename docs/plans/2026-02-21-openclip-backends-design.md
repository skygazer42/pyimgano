# OpenCLIP Backend + CLIP-based Detectors (PatchKNN + PromptScore) — Design

**Date:** 2026-02-21  
**Status:** Approved (Approach A selected)  
**Owner:** @codex (with user approval)

## Background

`pyimgano` has recently gained:

- Optional `anomalib` checkpoint wrappers (`vision_*_anomalib`) for fast access to strong baselines.
- A few-shot-friendly foundation-style detector `vision_anomalydino` (DINOv2 patch embeddings + kNN memory bank).

However, CLIP-based anomaly detection remains under-served:

- The current `winclip` implementation depends on OpenAI’s `clip` GitHub package (not pip-friendly).
- Many modern “CLIP-family” anomaly detection ideas can be approximated well in a **lightweight** way:
  - prompt-based scoring (zero-shot)
  - patch-token scoring + anomaly maps (pixel-level)
  - kNN memory bank on patch embeddings (few-shot)

This design adds a pip-installable, optional backend via **OpenCLIP** (`open_clip_torch`) and ships two
production-friendly detectors that fit `pyimgano`’s unified API.

## Goals

1. **Pip-friendly CLIP backend**
   - Add optional extra `pyimgano[clip]` that installs OpenCLIP (`open_clip_torch`).
   - Avoid Git-only installs for core CLIP functionality.
2. **Two CLIP-based detectors with unified semantics**
   - `decision_function(X)` returns continuous anomaly scores.
   - `predict(X)` returns 0/1 labels (after `fit()` calibrates a threshold).
   - `get_anomaly_map(path)` / `predict_anomaly_map(X)` provide pixel-level outputs where possible.
3. **Strong “default is strong” behavior for industrial AD**
   - Support image-level and pixel-level evaluation via existing pipeline helpers.
   - Keep few-shot usage simple (memory bank from normal/reference images).
4. **Offline/CI friendly**
   - No heavy imports at module import time.
   - Unit tests must run without OpenCLIP installed (injectable embedders).

## Non-Goals

- Reproducing AnomalyCLIP / AdaCLIP / AA-CLIP papers exactly (training recipes, losses, specialized data pipelines).
- Bundling pretrained weights inside `pyimgano`.
- Implementing a full text-prompt engineering framework.

## Proposed Public API

### New optional extra

- `pyimgano[clip]` → installs OpenCLIP via `open_clip_torch`

### New model registry entries

1. `vision_openclip_patchknn`
   - Patch embedding + kNN memory bank detector.
   - Implemented as an **adapter** around `vision_anomalydino`:
     - same score aggregation (`topk_mean` default)
     - same anomaly-map upsampling behavior
2. `vision_openclip_promptscore`
   - Prompt-based detector:
     - computes patch-level anomaly scores from similarity to “normal vs anomaly” text prompts
     - aggregates patch scores to image-level score
     - produces an anomaly map from patch-grid scores

## Dependency Strategy

- Add `[project.optional-dependencies] clip = ["open_clip_torch>=<min>"]`.
- Gate imports via `pyimgano.utils.optional_deps.require("open_clip", extra="clip", ...)`.
- Keep the rest of `pyimgano` core dependencies unchanged (OpenCLIP remains optional).

## Architecture

### A) OpenCLIP patch embedder (lazy)

Create a patch embedder implementing `VisionAnomalyDINO`’s `PatchEmbedder` protocol:

- `embed(image_path) -> (patch_embeddings, grid_shape, original_size)`
- Loads:
  - OpenCLIP model + preprocessing transforms
  - chosen ViT backbone + pretrained weights
- Extracts patch tokens robustly across OpenCLIP versions:
  - prefer “return tokens” / “output_tokens” APIs when available
  - otherwise raise a clear error requesting a custom embedder

### B) `vision_openclip_patchknn`

Implementation strategy:

- Thin wrapper around `VisionAnomalyDINO` using the OpenCLIP patch embedder by default.
- Keeps identical training + threshold calibration behavior:
  - `fit(train_paths)` builds memory bank and sets `threshold_` via `contamination` quantile.

### C) `vision_openclip_promptscore`

Core steps:

1. Encode text prompts into normalized text embeddings:
   - `normal` prompts and `anomaly` prompts (averaged within each group)
2. Encode image into normalized patch embeddings.
3. Compute patch anomaly score per patch, e.g.:
   - score = `sim(anomaly) - sim(normal)` (or ratio; finalized during implementation)
4. Aggregate patch scores to image score using `topk_mean` (same as AnomalyDINO default).
5. Reshape patch scores into `(grid_h, grid_w)` and upsample to `(H, W)` anomaly map.

Thresholding:

- `fit(train_paths)` calibrates a threshold from train scores.
- `predict(paths)` compares to `threshold_`.

## Error Handling

- Missing `open_clip` → `ImportError` with install hint: `pip install 'pyimgano[clip]'`.
- Missing anomaly map extraction support → actionable `RuntimeError` telling the user to:
  - upgrade OpenCLIP
  - or provide a custom embedder
- Empty training set → `ValueError` with clear message.
- Invalid `contamination` → `ValueError` (same convention as other backends).

## Testing Strategy

Unit tests must not require OpenCLIP:

- For `vision_openclip_patchknn`:
  - inject a fake patch embedder returning deterministic patch embeddings
  - verify `fit/decision_function/predict/get_anomaly_map` shapes and thresholding
- For `vision_openclip_promptscore`:
  - split core scoring into small numpy helpers (testable without torch/OpenCLIP)
  - inject fake patch embeddings + fake text embeddings to validate:
    - anomaly images score higher than normal
    - anomaly maps have correct shape and finite values

Optional (skip-if-missing) integration tests:

- If OpenCLIP is installed, smoke test that patch token extraction works for at least one ViT.

## Documentation

- Update `README.md` and `docs/DEEP_LEARNING_MODELS.md`:
  - how to install `pyimgano[clip]`
  - how to run `vision_openclip_*` detectors
  - note score vs label semantics (`decision_function` vs `predict`)
- Add an example script demonstrating:
  - fitting on normal images
  - evaluating image-level + pixel-level metrics via existing pipeline helpers

## Rollout / Work Plan

Implementation plan will be tracked in:

- `docs/plans/2026-02-21-openclip-backends.md`

