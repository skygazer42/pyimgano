# Industrial Pixel-First Upgrade (MVTec/VisA) — Design

**Date:** 2026-02-21  
**Status:** Approved  
**Owner:** @codex (with user approval)

## Background

`pyimgano` already provides many anomaly detection algorithms and a growing set of
industrial-focused utilities:

- Path-based dataset loaders (MVTec / VisA / BTAD).
- Pixel-level metrics (pixel AUROC / pixel AP / AUPRO).
- A MVTec/VisA pipeline (`pyimgano.pipelines.mvtec_visa`) that can align predicted
  anomaly maps to GT masks and compute pixel metrics.
- A “foundation-style” patch kNN model (`vision_anomalydino`) and OpenCLIP patch kNN
  backend (`vision_openclip_patchknn`) that already produce pixel-level anomaly maps.
- An inference-first `anomalib` checkpoint wrapper backend to quickly access other
  implementations without bloating core dependencies.

The remaining gaps for the **industrial visual inspection** scenario are:

1. **Pixel-first consistency** across detectors (maps, shapes, post-processing, reporting).
2. A stronger **robust patch-memory baseline** for noisy “normal” training data
   (common in production inspection lines).
3. Better **engineering ergonomics**: CLI parity with pipeline logic, clearer docs,
   and more examples focused on MVTec/VisA-style workflows.

## Goals

1. Make `pyimgano` “default-strong” for **pixel-first industrial AD**:
   - Pixel-level metrics are first-class (pixel AUROC / pixel AP / AUPRO).
   - Image-level scores remain available and consistent.
2. Keep the **core install lightweight**:
   - Any new heavy dependencies must be optional extras.
   - Optional dependencies must be imported lazily at runtime (not at `import pyimgano.models`).
3. Add at least one **robust patch-memory** detector inspired by modern industrial AD
   baselines (SoftPatch-style), with:
   - Training-time “normal noise” suppression / robust memory bank building.
   - Image-level scores + pixel-level anomaly maps.
4. Improve the **pipeline/CLI/docs** experience so that:
   - CLI uses the same pixel-map alignment logic as the pipeline.
   - It’s easy to benchmark on MVTec/VisA, save JSON reports, and export anomaly maps.

## Non-Goals

- Re-implement every SOTA training recipe from papers.
- Bundle pretrained weights into the package.
- Make `anomalib` a mandatory dependency (it stays optional).

## Proposed Approach (Architecture)

### A) Pixel-first pipeline as the “source of truth”

Keep `pyimgano.pipelines.mvtec_visa.evaluate_split()` as the canonical pixel-first
evaluation:

- Always compute image-level `scores = detector.decision_function(test_paths)`.
- If GT masks are present and the detector supports maps:
  - Use `predict_anomaly_map(paths)` when available; otherwise fall back to `get_anomaly_map(path)`.
  - Resize predicted maps to GT size.
  - Optionally run `AnomalyMapPostprocess` (normalize, blur, morphology, component filtering).
- Feed pixel labels/scores into `pyimgano.evaluation.evaluate_detector()` to compute
  pixel AUROC/AP/AUPRO.

The CLI should call into this pipeline logic (instead of re-implementing per-detector
map extraction).

### B) Patch-kNN “core helpers” for consistent scoring and maps

We already have reusable patch scoring helpers inside `pyimgano.models.anomalydino`.
This design extracts them into a small “core helpers” module so multiple patch-token
detectors can reuse the same:

- patch-score aggregation (`max` / `mean` / `topk_mean`)
- reshape patch scores into a grid
- safe upsampling from patch-grid to original image size

This improves consistency between:

- `vision_anomalydino`
- `vision_openclip_patchknn` (which wraps `vision_anomalydino`)
- future patch-token detectors (SoftPatch-like, other backbones)

### C) Add a robust patch-memory detector (SoftPatch-inspired)

Add a new detector focused on industrial pixel localization robustness:

`vision_softpatch` (name tentative)

High-level idea:

1. Extract patch embeddings (same interface style as PatchCore/AnomalyDINO).
2. During `fit(train_paths)`:
   - Estimate which training patches look like “contamination” / outliers.
   - Down-weight or remove those patches before building the memory bank.
3. During inference:
   - Score each patch by distance to the memory bank.
   - Upsample patch scores into an anomaly map and aggregate into an image-level score.

Implementation detail: keep dependencies minimal (NumPy + torch/torchvision already in core).
Any accelerator (FAISS) remains optional via `pyimgano[faiss]`.

### D) Expand `anomalib_backend` aliases for industrial models

`pyimgano.models.anomalib_backend` already provides inference-first wrappers. Expand alias
registry entries for models that are commonly used in industrial inspection and produce
pixel-level maps, for example:

- `vision_dinomaly_anomalib`
- `vision_cfa_anomalib`

These aliases are still the same underlying wrapper: users provide a checkpoint exported
by `anomalib`, and `pyimgano` handles inference + score/map extraction + threshold
calibration on a normal train split.

### E) Post-processing improvements (optional, safe defaults)

Enhance `AnomalyMapPostprocess` to support robust normalization options commonly used
in industrial AD pipelines:

- per-image percentile normalization (e.g. 1%–99%) to reduce the impact of extreme values
- optional clipping

This should remain opt-in via parameters and never silently change semantics for existing
users unless they choose it.

## Error Handling and Optional Dependencies

- Optional dependencies (`anomalib`, `faiss`, future heavy libs) must be checked with
  `pyimgano.utils.optional_deps.require()` at construction time, never at import time.
- For pixel maps:
  - If a detector returns maps with inconsistent shapes across images, pipeline evaluation
    should gracefully skip pixel metrics (already best-effort) or raise a clear error when
    the user explicitly asks for maps.

## Testing Strategy

1. Pure-numpy “core helpers” tests:
   - aggregation correctness and edge cases
   - reshape validation
2. `vision_anomalydino` regression tests:
   - still works with injected embedder (no torch required in unit tests)
   - optional coreset reduces memory bank size deterministically with seeded randomness
3. `vision_softpatch` tests:
   - injectable embedder for unit tests
   - robust memory filtering changes memory bank composition as expected on synthetic data
4. Pipeline/CLI parity:
   - CLI uses pipeline evaluation for pixel metrics (smoke test)

## References (web / GitHub)

- PatchCore official implementation: https://github.com/amazon-science/patchcore-inspection
- SoftPatch (robust patch memory): https://github.com/TencentYoutuResearch/AnomalyDetection-SoftPatch
- MSFlow (multi-scale flow): https://github.com/cool-xuan/msflow
- AnomalyDINO: https://github.com/dammsi/AnomalyDINO
- anomalib “Dinomaly” reference docs: https://anomalib.readthedocs.io/en/v2.1.0/markdown/guides/reference/models/image/dinomaly.html

