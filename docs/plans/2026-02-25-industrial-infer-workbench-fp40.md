# Industrial Infer + Workbench “FP40” Hardening — Implementation Plan (40 Tasks)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce industrial false positives and make `pyimgano-infer --defects` outputs (mask + regions) more stable and controllable, while improving inference throughput and strengthening workbench `infer_config.json` as a deployable, auditable artifact — without breaking existing CLIs or JSONL semantics.

**Architecture:** Extend the existing defects pipeline (`pyimgano.defects.*`) with opt-in FP controls (border suppression, smoothing, hysteresis, shape filters, deterministic ordering). Harden `pyimgano-infer` I/O (streaming JSONL, chunking/batching, optional profiling). Strengthen workbench/export contracts with additive schema versioning, validation, and provenance.

**Tech Stack:** Python, NumPy, OpenCV (cv2), Pillow, JSON/JSONL, pytest; optional torch for AMP (best-effort).

---

## Locked-in Constraints (User Requirements)

- **Backward compatible only**: add flags/fields/keys; do not change meanings of existing ones.
- **No service layer**: offline CLI tools only.
- **Release cadence** (4 milestones):
  - After tasks **#10 / #20 / #30 / #40**:
    - bump version
    - update `CHANGELOG.md`
    - create and push tag (`v0.6.18`, `v0.6.19`, `v0.6.20`, `v0.6.21`)

## Reference

- Design doc (already committed): `docs/plans/2026-02-25-industrial-infer-workbench-fp40-design.md`
- Defects pipeline:
  - `pyimgano/defects/extract.py`
  - `pyimgano/defects/binary_postprocess.py`
  - `pyimgano/defects/regions.py`
  - `pyimgano/defects/map_ops.py`
  - `pyimgano/defects/roi.py`
- Inference CLI: `pyimgano/infer_cli.py`
- Workbench config: `pyimgano/workbench/config.py`

---

## Milestone 1 (Tasks 1–10): Defects false-positive controls (release `v0.6.18`)

### Task 1: Add defect border suppression (`defects.border_ignore_px`)

**Files:**
- Modify: `pyimgano/defects/map_ops.py`
- Modify: `pyimgano/defects/extract.py`
- Modify: `pyimgano/workbench/config.py`
- Modify: `pyimgano/infer_cli.py`
- Test: `tests/test_defects_extract.py`
- Docs: `docs/CLI_REFERENCE.md`

**Steps:**
- Add a helper to zero out `N` pixels at the anomaly-map border (N=0 → no-op).
- Plumb `border_ignore_px` through `extract_defects_from_anomaly_map(...)`.
- Add workbench config key: `defects.border_ignore_px` (default 0).
- Add CLI flag: `--defect-border-ignore-px` (default 0).
- Add tests verifying border pixels are removed from the mask/regions when set.

### Task 2: Add anomaly-map smoothing before threshold (`defects.map_smoothing`)

**Files:**
- Create: `pyimgano/defects/smoothing.py`
- Modify: `pyimgano/defects/extract.py`
- Modify: `pyimgano/workbench/config.py`
- Modify: `pyimgano/infer_cli.py`
- Test: `tests/test_defects_extract.py`
- Docs: `docs/INDUSTRIAL_INFERENCE.md`

**Steps:**
- Implement `smooth_anomaly_map(map, method, ksize, sigma)` with method:
  - `none` (default), `median`, `gaussian`, `box`
- Apply smoothing only when enabled (config/flag).
- Add workbench config block:
  - `defects.map_smoothing.method` (default `"none"`)
  - `defects.map_smoothing.ksize` (default `0`)
  - `defects.map_smoothing.sigma` (default `0.0`)
- Add CLI flags:
  - `--defect-map-smoothing METHOD`
  - `--defect-map-smoothing-ksize INT`
  - `--defect-map-smoothing-sigma FLOAT`
- Tests: smoothing changes mask outcome on a synthetic noisy map.

### Task 3: Add hysteresis thresholding (`defects.hysteresis`)

**Files:**
- Create: `pyimgano/defects/hysteresis.py`
- Modify: `pyimgano/defects/extract.py`
- Modify: `pyimgano/workbench/config.py`
- Modify: `pyimgano/infer_cli.py`
- Test: `tests/test_defects_extract.py`
- Docs: `docs/INDUSTRIAL_INFERENCE.md`

**Steps:**
- Implement `hysteresis_mask(map, low, high)`:
  - compute low mask (>=low)
  - compute high seeds (>=high)
  - keep only low components that contain at least one high pixel
- Integrate in defects extraction as an alternative to single threshold.
- Config keys:
  - `defects.hysteresis.enabled` (default false)
  - `defects.hysteresis.low` / `high` (defaults derived from pixel_threshold unless explicitly set)
- CLI flags:
  - `--defect-hysteresis`
  - `--defect-hysteresis-low FLOAT`
  - `--defect-hysteresis-high FLOAT`

### Task 4: Add region shape filters (aspect/compactness/solidity)

**Files:**
- Modify: `pyimgano/defects/regions.py`
- Modify: `pyimgano/defects/extract.py`
- Modify: `pyimgano/workbench/config.py`
- Modify: `pyimgano/infer_cli.py`
- Test: `tests/test_defects_regions.py`
- Docs: `docs/CLI_REFERENCE.md`

**Steps:**
- Compute extra per-region stats (only when enabled):
  - `bbox_area`, `fill_ratio = area / bbox_area`
  - `aspect_ratio = max(w/h, h/w)`
  - optional `solidity` (area / convex_hull_area) when contour is available
- Add filters:
  - `defects.shape_filters.min_fill_ratio`
  - `defects.shape_filters.max_aspect_ratio`
  - `defects.shape_filters.min_solidity`
- Ensure filtering happens consistently with mask/regions expectations:
  - either filter at mask stage (preferred) or at regions stage with clear docs

### Task 5: Add region merge (nearby fragments) for regions output

**Files:**
- Create: `pyimgano/defects/merge.py`
- Modify: `pyimgano/defects/extract.py`
- Modify: `pyimgano/workbench/config.py`
- Modify: `pyimgano/infer_cli.py`
- Test: `tests/test_defects_extract.py`
- Docs: `docs/INDUSTRIAL_INFERENCE.md`

**Steps:**
- Implement best-effort region merging by bbox distance / centroid distance.
- Keep mask unchanged by default; merge affects regions list when enabled.
- Config keys:
  - `defects.merge_nearby.enabled` (default false)
  - `defects.merge_nearby.max_gap_px` (default 0)

### Task 6: Deterministic region ordering and tie-breakers

**Files:**
- Modify: `pyimgano/defects/extract.py`
- Test: `tests/test_defects_extract.py`

**Steps:**
- Ensure sort key is fully deterministic (score_max, area, bbox coords, centroid).
- Add tests verifying stable ordering across repeated calls.

### Task 7: Stable `max_regions` selection strategy (topK by configurable score)

**Files:**
- Modify: `pyimgano/defects/extract.py`
- Modify: `pyimgano/workbench/config.py`
- Modify: `pyimgano/infer_cli.py`
- Test: `tests/test_defects_extract.py`

**Steps:**
- Add `defects.max_regions_sort_by: score_max|score_mean|area` (default: score_max).
- Add CLI flag `--defect-max-regions-sort-by`.
- Ensure `max_regions` slices after sorting.

### Task 8: Optional image-space bbox mapping in JSONL

**Files:**
- Modify: `pyimgano/infer_cli.py`
- Create: `pyimgano/defects/space.py`
- Test: `tests/test_infer_cli_smoke.py`
- Docs: `docs/CLI_REFERENCE.md`

**Steps:**
- Add optional field in regions:
  - `bbox_xyxy_image` when `--defects-image-space` is enabled and image size can be read.
- Implement map→image scaling using:
  - anomaly-map `H,W`
  - original image `H,W` via Pillow `Image.open(...).size`
- Add CLI flag `--defects-image-space` (default false).

### Task 9: Add overlay export (`--save-overlays`) for FP debugging

**Files:**
- Create: `pyimgano/defects/overlays.py`
- Modify: `pyimgano/infer_cli.py`
- Test: `tests/test_infer_cli_smoke.py`
- Docs: `docs/INDUSTRIAL_INFERENCE.md`

**Steps:**
- When enabled, save an RGB overlay image per input:
  - original image
  - colormap anomaly heatmap (if map available)
  - binary mask outline fill (if defects enabled)
- Keep it opt-in: `--save-overlays DIR`.

### Task 10: Release `v0.6.18`

**Files:**
- Modify: `pyproject.toml`
- Modify: `pyimgano/__init__.py`
- Modify: `CHANGELOG.md`
- Modify: `examples/configs/industrial_adapt_defects_roi.json` (or add a new template)
- Docs: `docs/INDUSTRIAL_INFERENCE.md`

**Steps:**
- Add an FP-reduction example config template (ROI + smoothing/hysteresis).
- Bump version → `0.6.18`, update changelog.
- Tag: `git tag v0.6.18 && git push --tags`.

---

## Milestone 2 (Tasks 11–20): Throughput & profiling (release `v0.6.19`)

### Task 11: Stream JSONL writing (avoid building `records` list)

**Files:**
- Modify: `pyimgano/infer_cli.py`
- Test: `tests/test_infer_cli_smoke.py`

**Steps:**
- Refactor to write each record as it is produced.
- Keep stdout behavior identical when `--save-jsonl` is absent.

### Task 12: Add `--batch-size` (chunked inference)

**Files:**
- Modify: `pyimgano/infer_cli.py`
- Modify: `pyimgano/inference/api.py`
- Test: `tests/test_inference_api.py`
- Test: `tests/test_infer_cli_smoke.py`

**Steps:**
- Implement optional chunking in `infer(...)`:
  - process inputs in chunks of size N
  - preserve output order
- Add CLI flag `--batch-size` (default: 0/None meaning “auto/whole list”).

### Task 13: Stream map saving (avoid storing all `map_paths`)

**Files:**
- Modify: `pyimgano/infer_cli.py`
- Test: `tests/test_infer_cli_smoke.py`

**Steps:**
- Save maps per-record while streaming JSONL.
- Preserve current filenames and schema.

### Task 14: Add `--profile` stage timing (stderr summary)

**Files:**
- Modify: `pyimgano/infer_cli.py`
- Docs: `docs/CLI_REFERENCE.md`

**Steps:**
- Add `--profile` flag.
- Print stage timings to stderr (load model, fit/calibrate, infer, save artifacts).

### Task 15: Best-effort AMP/autocast (`--amp`)

**Files:**
- Modify: `pyimgano/inference/api.py`
- Modify: `pyimgano/infer_cli.py`
- Test: `tests/test_inference_api.py`
- Docs: `docs/INDUSTRIAL_INFERENCE.md`

**Steps:**
- Add `--amp` flag (default false).
- Wrap calls in `torch.inference_mode()` + `torch.cuda.amp.autocast` when torch available.
- Keep behavior best-effort: if torch missing, warn and continue without AMP.

### Task 16: Add `--include-anomaly-map-values` guard for debugging only

**Files:**
- Modify: `pyimgano/infer_cli.py`
- Modify: `pyimgano/inference/api.py`
- Test: `tests/test_infer_cli_smoke.py`

**Steps:**
- Expose `result_to_jsonable(..., include_anomaly_map_values=True)` behind a CLI flag.
- Ensure default remains off (large JSONL otherwise).

### Task 17: Optional mask compression (npz) (opt-in)

**Files:**
- Modify: `pyimgano/defects/io.py`
- Modify: `pyimgano/infer_cli.py`
- Test: `tests/test_defects_io.py`

**Steps:**
- Add format `npz` for masks (compressed numpy).
- Extend `--mask-format` choices to include `npz`.
- Keep defaults unchanged (`png`).

### Task 18: Improve tiling reuse (avoid recomputing coords)

**Files:**
- Modify: `pyimgano/inference/tiling.py`
- Test: `tests/test_inference_tiling.py`

**Steps:**
- Cache tile coordinates per (H,W,tile,stride) within `TiledDetector`.
- Add a regression test asserting identical stitched output.

### Task 19: Document tile seam blending knobs (already present)

**Files:**
- Docs: `docs/INDUSTRIAL_INFERENCE.md`
- Docs: `docs/CLI_REFERENCE.md`

**Steps:**
- Document `--tile-map-reduce max|mean|hann|gaussian` and when to use it.

### Task 20: Release `v0.6.19`

**Files:**
- Modify: `pyproject.toml`
- Modify: `pyimgano/__init__.py`
- Modify: `CHANGELOG.md`

**Steps:**
- Bump version → `0.6.19`, update changelog.
- Tag: `git tag v0.6.19 && git push --tags`.

---

## Milestone 3 (Tasks 21–30): Infer-config delivery & audit (release `v0.6.20`)

### Task 21: Add `infer_config.schema_version` (additive)

**Files:**
- Modify: `pyimgano/workbench/runner.py` (export payload)
- Modify: `pyimgano/infer_cli.py` (load payload)
- Test: `tests/test_infer_config_loader.py`

### Task 22: Infer-config validation (checkpoint path, defects keys types)

**Files:**
- Create: `pyimgano/inference/validate_infer_config.py`
- Test: `tests/test_infer_config_loader.py`

### Task 23: Add CLI `pyimgano-validate-infer-config`

**Files:**
- Create: `pyimgano/validate_infer_config_cli.py`
- Modify: `pyproject.toml` (`[project.scripts]`)
- Test: `tests/test_cli_smoke.py`
- Docs: `docs/CLI_REFERENCE.md`

### Task 24: Enrich pixel-threshold provenance (add counts + ROI info)

**Files:**
- Modify: `pyimgano/defects/pixel_threshold.py`
- Test: `tests/test_defects_pixel_threshold.py`

### Task 25: Enrich score threshold provenance (ensure parity)

**Files:**
- Modify: `pyimgano/calibration/score_threshold.py`
- Modify: `pyimgano/workbench/runner.py`
- Test: `tests/test_calibration_score_threshold.py`

### Task 26: Add `--seed` to `pyimgano-infer` (opt-in reproducibility)

**Files:**
- Modify: `pyimgano/infer_cli.py`
- Modify: `pyimgano/utils/experiment_tracker.py` (or central seed helper)
- Test: `tests/test_infer_cli_smoke.py`

### Task 27: Export optional deploy bundle (config + checkpoint + metadata)

**Files:**
- Modify: `pyimgano/train_cli.py`
- Modify: `pyimgano/workbench/runner.py`
- Docs: `docs/WORKBENCH.md`

### Task 28: Add infer-config completeness checks for defects export

**Files:**
- Modify: `pyimgano/workbench/runner.py`
- Modify: `pyimgano/workbench/config.py`
- Test: `tests/test_examples_configs_load.py`

### Task 29: Ensure infer uses exported preprocess/tiling defaults consistently

**Files:**
- Modify: `pyimgano/infer_cli.py`
- Test: `tests/test_integration_workbench_train_then_infer.py`

### Task 30: Release `v0.6.20`

**Files:**
- Modify: `pyproject.toml`
- Modify: `pyimgano/__init__.py`
- Modify: `CHANGELOG.md`

**Steps:**
- Bump version → `0.6.20`, update changelog.
- Tag: `git tag v0.6.20 && git push --tags`.

---

## Milestone 4 (Tasks 31–40): Ecosystem alignment + recipes + docs (release `v0.6.21`)

### Task 31: Strengthen sklearn adapter (input validation + errors)

**Files:**
- Modify: `pyimgano/sklearn_adapter.py`
- Test: `tests/test_detectors_compat.py`
- Docs: `docs/SKLEARN_INTEGRATION.md`

### Task 32: Expand detector contract tests set

**Files:**
- Modify: `tests/contracts/test_detector_contract.py`

**Steps:**
- Add representative pixel-map detector(s) if feasible (gated by optional deps).
- Add at least one deep detector smoke test with paths inputs.

### Task 33: Dataset API guidance (“recommended paths”)

**Files:**
- Docs: `docs/QUICKSTART.md`
- Docs: `docs/README_DOCS.md`

### Task 34: Add 2–3 industrial recipes (existing models only)

**Files:**
- Modify/Create: `pyimgano/recipes/builtin/*.py`
- Docs: `docs/RECIPES.md`
- Test: `tests/test_integration.py` (smoke)

### Task 35: Add false-positive debugging guide

**Files:**
- Create: `docs/FALSE_POSITIVE_DEBUGGING.md`
- Docs: `docs/INDUSTRIAL_INFERENCE.md` (link)

### Task 36: Add opt-in illumination/contrast preprocessing pipeline knobs

**Files:**
- Modify: `pyimgano/preprocessing/industrial_presets.py`
- Modify: `pyimgano/preprocessing/enhancer.py`
- Test: `tests/test_industrial_augmentation.py`
- Docs: `docs/PREPROCESSING.md`

### Task 37: Document “torch-like” extension points

**Files:**
- Docs: `docs/DEEP_LEARNING_MODELS.md`
- Docs: `docs/MODEL_INDEX.md`

### Task 38: Update comparison positioning (vs PyOD/anomalib)

**Files:**
- Modify: `docs/COMPARISON.md`

### Task 39: Update READMEs (EN + translations) for FP40 workflow

**Files:**
- Modify: `README.md`
- Modify: `README_cn.md`
- Modify: `README_ja.md`
- Modify: `README_ko.md`

### Task 40: Release `v0.6.21`

**Files:**
- Modify: `pyproject.toml`
- Modify: `pyimgano/__init__.py`
- Modify: `CHANGELOG.md`
- Docs: `docs/PUBLISHING.md`

**Steps:**
- Verify PyPI constraints (no direct URL deps in metadata).
- Bump version → `0.6.21`, update changelog.
- Tag: `git tag v0.6.21 && git push --tags`.

