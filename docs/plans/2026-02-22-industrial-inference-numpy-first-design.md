# Industrial Inference (numpy-first) + Explicit `ImageFormat` — Design

**Date:** 2026-02-22  
**Status:** Approved  
**Owner:** @codex (with user approval)

## Background

`pyimgano` already has a broad set of anomaly detection models (classical + deep) and an
industrial-focused benchmark pipeline (`pyimgano.pipelines.mvtec_visa`) that can compute
image-level and pixel-level metrics.

For **industrial deployment** and **embedding into a larger system**, the current biggest gap is
input/output ergonomics:

- Production systems often have frames already decoded in memory as **numpy arrays** (sometimes
  BGR/HWC/uint8, sometimes RGB/HWC/uint8, sometimes RGB/CHW/float32).
- “Auto-guessing” formats is fragile and silently harms accuracy.
- Different detectors accept different input types today (mostly paths); pixel maps are available
  only for some detectors.

The user requirement is:

- Support **path / numpy / torch** inputs overall, but **numpy-first**.
- Require **explicit input format** when consuming in-memory images.
- Output **image-level scores as the default**, with **pixel anomaly maps optional** when available.
- Keep core install lightweight; heavy deps remain optional and lazily imported.

## Goals

1. Provide a stable, production-friendly **inference SDK** that:
   - Accepts `np.ndarray` images with explicit `input_format`.
   - Returns structured per-image results (score/label + optional anomaly map).
2. Establish a strict and testable **image input contract**:
   - No implicit guessing of BGR↔RGB or value range.
   - Deterministic conversions to a single internal canonical representation.
3. Incrementally upgrade the most relevant industrial detectors to support numpy input natively:
   - `vision_patchcore`, `vision_padim`, `vision_spade`, `vision_anomalydino`, `vision_softpatch`,
     `vision_stfpm`, `vision_reverse_distillation` (and alias), `vision_draem`.
4. Improve CLI ergonomics for industrial usage without breaking existing benchmark CLI:
   - Add an inference-oriented entrypoint (calibration + scoring + optional map export).
5. Add at least one **practical industrial “popular algorithm”** implemented in-framework:
   - A score-level ensemble wrapper (`vision_score_ensemble`) for robust accuracy improvements.

## Non-Goals

- Re-implement every SOTA training recipe.
- Bundle weights into the wheel.
- Add “magic auto-detect” of input image format.
- Force optional heavy dependencies into the core install.

## Proposed Architecture

### A) `pyimgano.inputs`: explicit `ImageFormat` and canonicalization

Add a small, dependency-light module `pyimgano.inputs`:

- `ImageFormat` (string enum), initially supporting:
  - `bgr_u8_hwc` (OpenCV-like)
  - `rgb_u8_hwc` (PIL-like)
  - `rgb_f32_chw` (torch-like, value range [0, 1])
- `normalize_numpy_image(image, *, input_format) -> np.ndarray`
  - Output is canonical: **RGB / HWC / uint8 / [0,255]**.
  - Strict validation; clear errors when shape/dtype don’t match the declared format.
- Optional helpers:
  - `ensure_rgb_u8_hwc(image)` for detector-level guardrails.
  - `to_torch_chw_float(image_rgb_u8_hwc, normalize="imagenet"|None)`.

Rationale: keeping one canonical representation reduces ambiguity and makes detector upgrades
incremental and testable.

### B) `pyimgano.inference`: stable, embedding-friendly API

Add a user-facing inference layer (thin wrapper around detectors):

- `calibrate_threshold(detector, normal_images, *, input_format, contamination=None)`
  - Runs `decision_function` on normal data and sets/returns a threshold.
  - Uses detector-native thresholding when available (`threshold_` / `decision_scores_`), otherwise
    uses a generic quantile rule.
- `infer(detector, images, *, input_format, return_maps=False) -> list[InferenceResult]`
  - Returns `score` always.
  - Returns `label` when a threshold is available (or after calibration).
  - Returns `anomaly_map` only when `return_maps=True` and the detector supports maps.

This layer is where production systems integrate; the CLI becomes a small wrapper around it.

### C) Model “capabilities” and discoverability

Codify which detectors support which capabilities via registry metadata and/or tags:

- `supports_numpy_input`
- `supports_pixel_map`
- `requires_checkpoint`

This enables:

- Better CLI UX (list models by capability/tags).
- Better error messages (“this detector is path-only; use `pyimgano.inputs.normalize_*` + detector X
  or choose an alternative detector”).

### D) Numpy support strategy for detectors

We avoid an all-at-once refactor. Instead:

1. Define canonical numpy representation (RGB/u8/HWC).
2. Upgrade high-ROI industrial detectors to accept:
   - `Iterable[str]` (unchanged)
   - `Iterable[np.ndarray]` (canonical; validated)
   - In some cases `np.ndarray` batch `(N,H,W,C)` as convenience
3. For “training-loop” detectors inheriting `BaseVisionDeepDetector`, add a dedicated array dataset
   (so they can run fit/inference without needing filesystem paths).

### E) Add “popular industrial” algorithm: `vision_score_ensemble`

Implement a pragmatic score-level ensemble wrapper:

- Wrap multiple already-fitted detectors.
- Normalize their scores based on their calibration distribution (`decision_scores_` or provided
  calibration set).
- Combine by mean/median/rank-mean.

Rationale: ensembles are widely used in production because they improve robustness without
requiring new training pipelines.

### F) CLI: inference-oriented entrypoint

Add a new CLI command (new script entrypoint) such as:

- `pyimgano-infer`

Capabilities:

- `--model`, `--preset`, `--model-kwargs`, `--checkpoint-path`
- `--train-dir` (optional): normal images for threshold calibration
- `--input` (file/dir/glob): images to score
- `--save-jsonl` (default): one JSON record per image
- `--save-maps` (optional): export anomaly maps when supported

This CLI is intentionally “industrial”: it optimizes for stable machine-readable output and easy
integration into other systems, not for benchmark reporting.

## Error Handling

- `ImageFormat` conversion errors must be explicit and actionable:
  - “Declared `bgr_u8_hwc` but got dtype=float32…”
  - “Declared `rgb_f32_chw` but got shape (H,W,C)…”
- If a detector does not support maps:
  - `infer(..., return_maps=True)` returns `anomaly_map=None` (or raises only if user requests a
    strict mode; default is best-effort).
- Optional dependencies remain lazy:
  - Any heavy library import must happen at construction-time or call-time with
    `pyimgano.utils.optional_deps.require()`.

## Testing Strategy

1. **Pure input contract tests** (`pyimgano.inputs`):
   - parse/validation for supported formats
   - canonicalization correctness (channel swaps, CHW↔HWC, float↔uint8)
2. **Inference API tests** (`pyimgano.inference`):
   - works for score-only detectors
   - works for map-capable detectors
   - threshold calibration behavior is deterministic
3. **Detector upgrades**:
   - for each upgraded industrial detector, add at least one unit test that exercises numpy input
     without filesystem paths (use synthetic arrays; keep torch usage minimal when possible)
4. **CLI smoke tests**:
   - monkeypatch `create_model` and use a temp directory with a few dummy images to verify JSONL
     output shape and error handling.

## Documentation & Examples

- Add an “Industrial Inference” section showing:
  - numpy-first usage with explicit `input_format`
  - calibration + inference loop
  - how to request anomaly maps when available
- Add examples:
  - `examples/industrial_infer_numpy.py`
  - `examples/industrial_score_ensemble.py`

## References

- SoftPatch repository (robust patch memory): https://github.com/TencentYoutuResearch/AnomalyDetection-SoftPatch
- PatchCore official implementation: https://github.com/amazon-science/patchcore-inspection
- anomalib deployment inferencer: https://anomalib.readthedocs.io/
