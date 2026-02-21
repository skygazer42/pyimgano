# anomalib Backend Expansion + AnomalyDINO PoC — Design

**Date:** 2026-02-21  
**Status:** Approved  
**Owner:** @codex (with user approval)

## Background

`pyimgano` already has many native anomaly detection implementations, but:

1. Adding / maintaining many SOTA implementations natively is costly.
2. The open-source ecosystem (especially Intel’s `anomalib`) has robust, widely used reference
   implementations and consistent outputs (image-level scores + pixel-level anomaly maps).
3. Recent “foundation model” approaches (e.g. DINOv2-based patch kNN) are attractive for **few-shot**
   anomaly detection on industrial datasets.

This design expands `pyimgano` with a **Hybrid** strategy:

- Keep `pyimgano` as the **stable API + datasets + eval/metrics + post-processing + reporting** layer.
- Add more models quickly via an **optional** `anomalib` backend wrapper.
- Add one “foundation-style” PoC model: **AnomalyDINO** (DINOv2 patch embeddings + kNN).

## Goals

1. **Add more algorithms quickly** without bloating `pyimgano` core:
   - Provide `vision_*_anomalib` wrappers for popular `anomalib` checkpoints.
2. **Unify semantics** across all detectors:
   - `decision_function(X)` returns continuous anomaly scores.
   - `predict(X)` returns binary labels (0/1) using a fitted or calibrated threshold.
   - Optional pixel-level outputs via `get_anomaly_map(path)` / `predict_anomaly_map(X)`.
3. **Foundation model PoC**:
   - Add `vision_anomalydino` implementing DINOv2 patch-embedding kNN scoring, supporting:
     - one-class / few-shot training on normal/reference images
     - image-level scores + pixel-level heatmaps
4. **Offline/CI friendly**:
   - Avoid mandatory weight downloads during import.
   - Keep torch-dependent code lazy (import at runtime), so unit tests can run in minimal environments.

## Non-Goals

- Re-implement all `anomalib` training pipelines inside `pyimgano`.
- Bundle pretrained weights into the package.
- Implement MuSc now (batch-dependent zero-shot scoring); we will consider it after AnomalyDINO lands.

## Proposed Architecture

### A) anomalib backend wrappers

#### Public API

- `vision_anomalib_checkpoint` (generic)
- Alias entries for convenience (all use the same implementation under the hood), e.g.:
  - `vision_patchcore_anomalib`
  - `vision_padim_anomalib`
  - `vision_stfpm_anomalib`
  - `vision_draem_anomalib`
  - `vision_fastflow_anomalib`
  - …

#### Semantics

- `fit(train_paths)` does not train (training is done via `anomalib`), but **calibrates**
  a threshold from scores on `train_paths`:
  - `self.decision_scores_ = decision_function(train_paths)`
  - `self.threshold_ = quantile(decision_scores_, 1 - contamination)`
- `decision_function(paths)` returns the `pred_score` / image-level score provided by anomalib’s inferencer.
- `get_anomaly_map(path)` returns `anomaly_map` from anomalib’s inferencer (converted to numpy).

#### Testability

To avoid requiring `anomalib` for unit tests, the wrapper supports injecting a fake inferencer.
Unit tests validate:
- score extraction from dict/object results
- threshold calibration path
- `predict()` outputs {0,1}
- `predict_anomaly_map()` stacks maps and converts to numpy

### B) `vision_anomalydino` PoC

#### Reference

- “AnomalyDINO: Boosting Patch-Based Few-Shot Anomaly Detection with DINOv2” (WACV 2025)
- Official repo: `dammsi/AnomalyDINO` (Apache-2.0)

#### Core idea

1. Extract **patch embeddings** from a DINOv2 ViT backbone.
2. Build a **memory bank** from patch embeddings of normal/reference images.
3. Score a test image by nearest-neighbor distance from each patch embedding to the memory bank.
4. Aggregate patch distances to image-level score (default: `topk_mean`).
5. Reshape patch scores into a patch-grid anomaly map and upsample to original resolution.

#### Implementation approach

Split into two layers:

1. **Core algorithm** (pure numpy):
   - memory bank + kNN search via `pyimgano.models.knn_index`
   - score aggregation + anomaly-map reshaping
   - fully testable without torch
2. **Default embedder** (torch.hub DINOv2):
   - loaded lazily at runtime
   - can be replaced by a user-provided embedder for offline/enterprise environments

#### Outputs

- `decision_function(paths)` → image-level scores
- `predict(paths)` → 0/1 labels via calibrated `threshold_`
- `get_anomaly_map(path)` and `predict_anomaly_map(paths)` → pixel anomaly heatmaps

## References (web)

- anomalib deployment inferencer outputs (score + anomaly map):  
  https://anomalib.readthedocs.io/en/latest/markdown/guides/reference/deploy/
- anomalib model index / catalog (for alias coverage):  
  https://anomalib.readthedocs.io/en/v1.2.0/genindex.html
- AnomalyDINO paper page:  
  https://openaccess.thecvf.com/content/WACV2025/html/Damm_AnomalyDINO_Boosting_Patch-Based_Few-Shot_Anomaly_Detection_with_DINOv2_WACV_2025_paper.html
- AnomalyDINO GitHub repo (Apache-2.0):  
  https://github.com/dammsi/AnomalyDINO

