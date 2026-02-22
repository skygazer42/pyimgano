# Industrial One-Click Pipeline (CLI + Python API) — Design

**Date:** 2026-02-22  
**Status:** Proposed (pending final approval)  
**Owner:** @codex

## Background

`pyimgano` already has strong building blocks for industrial anomaly detection:

- Registry-driven models (`pyimgano.models.create_model`)
- Standard benchmark datasets (MVTec AD / LOCO / AD2, VisA, BTAD) via `pyimgano.utils.datasets.load_dataset`
- A path-first benchmark pipeline (`pyimgano.pipelines.mvtec_visa`)
- Evaluation metrics including image-level + pixel-level + AUPRO (`pyimgano.evaluation`)
- CLI utilities:
  - `pyimgano-benchmark` (dataset evaluation)
  - `pyimgano-infer` (inference)
  - `pyimgano-robust-benchmark` (corruption robustness)

However, the “industrial workflow” is still missing a **one-click** entry point that:

1) Supports running a **single category** or **all categories** (`--category all`).  
2) Produces a standard **run artifact layout** on disk (summary + per-image JSONL).  
3) Uses an **industrial-safe threshold strategy** (train calibration) rather than test-label-optimized thresholds.

This design adds that without introducing heavy new dependencies or rewriting existing model code.

## Goals

### Must-have (user approved)

1. **CLI + Python API** both supported for the one-click pipeline.
2. **Datasets**: support standard benchmarks + a “custom dataset” loader.
3. **Category routing**:
   - `--category <name>` runs a single category
   - `--category all` runs every category and aggregates results
4. **Output artifacts (default)**:
   - `report.json` (run-level + per-category summary)
   - per-category `per_image.jsonl` (one record per test image)
5. **Thresholding (default)**:
   - image-level predicted labels use a **train-calibrated** score threshold
   - no “oracle” threshold optimized on test labels by default
6. **Output directory**:
   - default: `runs/<timestamp>_<dataset>_<model>/`
   - category=all writes per-category outputs under `categories/<cat>/...`

### Nice-to-have (safe extensions)

- Optional limits for quick runs (`--limit-train`, `--limit-test`)
- Optional ability to override calibration quantile (`--calibration-quantile`)
- Keep backward compatibility of `pyimgano-benchmark` stdout JSON output

## Non-Goals

- Rewriting model implementations or training loops.
- Bundling pretrained weights into the package.
- Adding visualization artifacts by default (maps/overlays stay optional).

## Public Interfaces

### 1) Python API

Add a new orchestration API in `pyimgano.pipelines`:

- `run_benchmark(...)`:
  - supports `category="all"` or single category
  - returns a run payload compatible with JSON serialization
  - optionally writes run artifacts to `output_dir`

The API is designed to be called from notebooks and production pipelines, while the CLI remains
the thin wrapper.

### 2) CLI (`pyimgano-benchmark`)

Extend `pyimgano-benchmark` with:

- `--dataset custom` support
- `--category all`
- `--output-dir` (optional; default auto)
- `--save-run/--no-save-run` (default: save)
- `--per-image-jsonl/--no-per-image-jsonl` (default: enabled when saving)
- `--resize H W`
- `--calibration-quantile` (optional score threshold quantile)
- `--limit-train`, `--limit-test`

**Backward compatibility:**
- Keep existing JSON output printed to stdout (tests and users rely on it).
- Keep `--output` behavior (still writes a copy of the summary report JSON).

## Output Layout

For `--category bottle`:

```
runs/<ts>_<dataset>_<model>/
  report.json
  config.json
  categories/
    bottle/
      report.json
      per_image.jsonl
```

For `--category all`:

```
runs/<ts>_<dataset>_<model>/
  report.json                   # includes per-category summaries + mean/std
  config.json
  categories/
    <cat1>/report.json
    <cat1>/per_image.jsonl
    <cat2>/report.json
    <cat2>/per_image.jsonl
    ...
```

Per-image JSONL record (minimal):

```json
{"index":0,"dataset":"mvtec","category":"bottle","input":".../test.png","y_true":0,"score":0.123,"threshold":0.456,"pred":0}
```

## Threshold Strategy (Industrial Default)

- Fit detector on `train_paths` (normal-only).
- Calibrate a score threshold from **train** scores:
  - default quantile: `1 - contamination` when contamination is available
  - otherwise fallback to `0.995`
- Use this threshold for:
  - `pred` in per-image outputs
  - classification metrics in the summary report

Threshold-free metrics (AUROC, AP) are unchanged.

## Risks / Mitigations

- **More files written by default**: add `runs/` to `.gitignore`.
- **Different “F1” compared to oracle thresholds**: this is intentional for industrial realism.
- **Dataset category discovery differences**: implement a centralized category listing helper for each dataset.

## Acceptance Criteria

1. `pyimgano-benchmark --dataset mvtec --root ... --category all --model vision_patchcore` works.
2. A run directory is created by default with the expected file layout.
3. `report.json` contains:
   - run metadata (dataset/model/preset/device/threshold strategy)
   - per-category metrics
   - overall mean metrics across categories
4. Per-category `per_image.jsonl` exists and includes `score`, `threshold`, `pred`, `y_true`.
5. Existing CLI tests continue to pass (stdout JSON still printed).

