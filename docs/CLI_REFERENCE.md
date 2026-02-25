# CLI Reference

PyImgAno provides five primary CLIs:

- `pyimgano-benchmark` — one-click industrial benchmarking + run artifacts
- `pyimgano-train` — recipe-driven workbench runs (adaptation-first; optional micro-finetune)
- `pyimgano-infer` — JSONL inference over images/videos (path-driven)
- `pyimgano-robust-benchmark` — robustness evaluation (clean + corruptions)
- `pyimgano-manifest` — generate a JSONL manifest from a `custom`-layout dataset tree

---

## `pyimgano-benchmark`

### Common Usage

```bash
pyimgano-benchmark \
  --dataset mvtec \
  --root /path/to/mvtec_ad \
  --category bottle \
  --model vision_patchcore \
  --preset industrial-balanced \
  --device cuda
```

### Discovery

- List models: `pyimgano-benchmark --list-models`
- Filter models by tags: `pyimgano-benchmark --list-models --tags vision,deep`
- Model info (constructor signature + accepted kwargs): `pyimgano-benchmark --model-info vision_patchcore`
- List dataset categories: `pyimgano-benchmark --list-categories --dataset mvtec --root /path/to/mvtec_ad`
- List manifest categories: `pyimgano-benchmark --list-categories --dataset manifest --manifest-path /path/to/manifest.jsonl`

### Manifest dataset

Benchmark a manifest JSONL (paths mode):

```bash
pyimgano-benchmark \
  --dataset manifest \
  --root /path/to/fallback_root \
  --manifest-path /path/to/manifest.jsonl \
  --category bottle \
  --model vision_patchcore \
  --device cuda
```

Optional split policy knobs:

- `--manifest-test-normal-fraction 0.2`
- `--manifest-split-seed 123` (defaults to `--seed` or 0)

### Run Artifacts

By default, run artifacts are written to `runs/<timestamp>_<dataset>_<model>/`:

- `report.json`
- `config.json`
- `environment.json`
- `categories/<category>/report.json`
- `categories/<category>/per_image.jsonl`

Key flags:

- `--output-dir DIR` — write artifacts to a specific directory
- `--save-run/--no-save-run` — enable/disable artifact writing
- `--per-image-jsonl/--no-per-image-jsonl` — enable/disable per-image JSONL

### Inputs

- `--input-mode paths|numpy`
  - `paths`: pass file paths to detectors (default)
  - `numpy`: decode images into memory first (for numpy-first models)

### Reproducibility

- `--seed INT` — best-effort deterministic seeding (also passed as `random_seed/random_state` when supported)

### Threshold Calibration

- Default strategy: calibrate `threshold_` as a quantile of **train/normal** scores.
- Default quantile: `1 - contamination` when available, else `0.995`.
- Override quantile: `--calibration-quantile Q`
- Run artifacts include `threshold_provenance` (quantile + where it came from).

### Model Persistence (Classical Detectors Only)

- Save detector after fit: `--save-detector [PATH|auto]`
  - `auto` writes `<output-dir>/detector.pkl` (or `runs/.../detector.pkl` if `--output-dir` is omitted)
- Load detector and skip fitting: `--load-detector PATH`

Security note: never load pickle files from untrusted sources.

### Feature Cache (Path Inputs, Classical Detectors)

- `--cache-dir DIR` — cache extracted feature vectors on disk (speeds up repeated scoring)

---

## `pyimgano-infer`

Runs inference and emits one JSON record per input (stdout or JSONL file).

Example:

```bash
pyimgano-infer \
  --model vision_padim \
  --train-dir /path/to/train/good \
  --input /path/to/inputs \
  --save-jsonl out.jsonl
```

Notes:

- When `--train-dir` is provided and the detector does **not** set `threshold_` during `fit()`,
  `pyimgano-infer` auto-calibrates `threshold_` from train scores (same default quantile as
  `pyimgano-benchmark`: `1 - contamination` when available, else `0.995`).
- Pass `--calibration-quantile Q` to override the quantile explicitly.

Optional:

- `--include-maps` + `--save-maps DIR` — write anomaly maps as `.npy`
- High-resolution tiling (optional; for 2K/4K inspection images):
  - `--tile-size N` — run tiled inference (wraps the detector in `TiledDetector`)
  - `--tile-stride N` — tile overlap stride (default: tile-size; smaller = more overlap = fewer seams)
  - `--tile-map-reduce max|mean|hann|gaussian` — blend overlapping tile maps (`hann`/`gaussian` reduce seams)
  - `--tile-score-reduce max|mean|topk_mean` + `--tile-score-topk` — aggregate tile scores into an image score
- `--batch-size N` — run inference in chunks (preserves output order; can reduce peak memory)
- `--profile` — print stage timing summary to stderr (load model, fit/calibrate, infer, artifacts)
- `--amp` — best-effort AMP/autocast for torch-backed models (requires torch + CUDA; otherwise runs without AMP)
- `--include-anomaly-map-values` — embed raw anomaly-map values in JSONL (debug only; very large output)
- `--defects` — export industrial defect structures (binary mask + connected-component regions)
  - `--save-masks DIR` + `--mask-format png|npy|npz` (`npz` is compressed numpy; good for large batches)
  - `--save-overlays DIR` — save per-image debugging overlays (original + heatmap + mask outline/fill)
  - `--defects-image-space` — add `bbox_xyxy_image` to regions when image size is available
  - Pixel threshold options:
    - `--pixel-threshold FLOAT` + `--pixel-threshold-strategy fixed`
	    - `--pixel-threshold-strategy infer_config` (uses `defects.pixel_threshold` from `infer_config.json` / a workbench run)
	    - `--pixel-threshold-strategy normal_pixel_quantile` (requires `--train-dir`; uses `--pixel-normal-quantile`)
	      - If selected and `--train-dir` is provided, `pyimgano-infer` recalibrates from normal/train maps even if `infer_config.json` contains `defects.pixel_threshold`.
  - When running with `--infer-config` or `--from-run`, the exported `defects.*` settings are used as defaults
    (ROI, morphology, min-area, mask format, max regions, pixel threshold strategy/quantile, etc.). CLI flags override.
  - `--roi-xyxy-norm x1 y1 x2 y2` (optional; gates defects output only)
    - If ROI is set and you calibrate pixel threshold via `normal_pixel_quantile`, calibration uses ROI pixels only.
  - `--defect-border-ignore-px INT` (optional; ignores N pixels at the anomaly-map border for defects extraction)
  - Map smoothing (optional; reduces speckle before thresholding):
    - `--defect-map-smoothing none|median|gaussian|box`
    - `--defect-map-smoothing-ksize INT`
    - `--defect-map-smoothing-sigma FLOAT` (gaussian only)
  - Hysteresis thresholding (optional; keeps low regions connected to high seeds):
    - `--defect-hysteresis`
    - `--defect-hysteresis-low FLOAT`
    - `--defect-hysteresis-high FLOAT`
  - Shape filters (optional; useful to remove long thin strips / speckle fragments):
    - `--defect-min-fill-ratio FLOAT` — drop components whose `area / bbox_area` is below this threshold
    - `--defect-max-aspect-ratio FLOAT` — drop components whose bbox aspect ratio exceeds this threshold
    - `--defect-min-solidity FLOAT` — drop components whose solidity (contour / convex hull) is below this threshold
  - Region merge (optional; affects regions list only, mask unchanged):
    - `--defect-merge-nearby`
    - `--defect-merge-nearby-max-gap-px INT`
  - Output limiting (optional):
    - `--defect-max-regions INT`
    - `--defect-max-regions-sort-by score_max|score_mean|area`
  - Region-level filters (optional):
    - `--defect-min-score-max FLOAT` — drop components whose max anomaly score is below the threshold
    - `--defect-min-score-mean FLOAT` — drop components whose mean anomaly score is below the threshold
- `--from-run RUN_DIR` — load model/threshold/checkpoint from a prior `pyimgano-train` workbench run
  - If the run contains multiple categories, pass `--from-run-category NAME`.
- `--infer-config PATH` — load model/threshold/checkpoint from an exported workbench infer-config
  - For example: `runs/.../artifacts/infer_config.json`
  - If the infer-config contains multiple categories, pass `--infer-category NAME`.

Defects export example:

```bash
pyimgano-infer \
  --model vision_patchcore \
  --train-dir /path/to/train/good \
  --input /path/to/inputs \
  --defects \
  --save-masks /tmp/pyimgano_masks \
  --mask-format png \
  --pixel-threshold 0.5 \
  --pixel-threshold-strategy fixed \
  --roi-xyxy-norm 0.1 0.1 0.9 0.9 \
  --save-jsonl out.jsonl
```

Each JSONL record includes a `defects` block when `--defects` is enabled:

```json
{
  "defects": {
    "pixel_threshold": 0.5,
    "pixel_threshold_provenance": {"method": "fixed", "source": "explicit"},
    "mask": {"path": "masks/000000_x.png", "shape": [256, 256], "dtype": "uint8", "encoding": "png"},
    "regions": [{"id": 1, "bbox_xyxy": [12, 34, 80, 120], "area": 1532, "centroid_xy": [45.2, 77.8]}]
  }
}
```

---

## `pyimgano-train`

Runs a **recipe-driven workbench** run from a JSON-first config file. This is the
recommended entrypoint for industrial “adaptation-first” workflows where you
want reproducible artifacts (config, environment snapshot, per-image JSONL, etc.).

See also: `docs/RECIPES.md`

### Discovery

- List recipes: `pyimgano-train --list-recipes`
- List recipes (JSON): `pyimgano-train --list-recipes --json`
- Recipe info: `pyimgano-train --recipe-info industrial-adapt`
- Recipe info (JSON): `pyimgano-train --recipe-info industrial-adapt --json`

### Run a recipe

```bash
pyimgano-train --config cfg.json
```

### Artifact layout

Workbench runs follow a benchmark-compatible layout and add extra folders:

```
<run_dir>/
  report.json
  config.json
  environment.json
  categories/<cat>/report.json
  categories/<cat>/per_image.jsonl
  checkpoints/...
  artifacts/...
```

### Common overrides

Flags override the config (useful for quick experiments):

- `--dataset NAME` / `--root PATH` / `--category CAT`
- `--model MODEL_NAME` / `--device cpu|cuda`

### Preflight dataset validation

Validate dataset health and emit a machine-readable JSON report (no training):

```bash
pyimgano-train --config cfg.json --preflight
```

Behavior:

- Prints: `{"preflight": ...}` (JSON) to stdout.
- Returns exit code `0` when no `severity="error"` issues exist.
- Returns exit code `2` when any `severity="error"` issue exists.

This is intended for CI and pipeline orchestration (e.g. detect missing files,
duplicate paths, manifest group split conflicts, or incomplete anomaly masks
before starting a recipe run).

### Notes

- Training-enabled workbench runs persist checkpoints under `checkpoints/<category>/...` when supported.
- For reusing a run in deploy-style inference, see `pyimgano-infer --from-run` and `docs/RECIPES.md`.
- For manifest datasets, `pyimgano-train --dry-run` validates that `dataset.manifest_path` exists and is readable.

---

## `pyimgano-robust-benchmark`

Runs clean + corruption robustness evaluation (when supported by the selected model/input mode).

Example:

```bash
pyimgano-robust-benchmark \
  --dataset mvtec \
  --root /path/to/mvtec_ad \
  --category bottle \
  --model vision_patchcore \
  --device cuda
```

---

## `pyimgano-manifest`

Generate a JSONL manifest for the built-in `custom` dataset layout.

Example:

```bash
pyimgano-manifest \
  --root /path/to/custom_dataset \
  --out /path/to/manifest.jsonl \
  --include-masks
```

Notes:

- Output is stable and sorted (useful for reproducible diffs).
- By default, paths are written relative to the output manifest directory.
- Use `--absolute-paths` to emit absolute paths when you need portability across working directories.

## Notes

- Many models have optional backends; when required dependencies are missing, error messages include install hints.
- For a full model catalog, see `docs/MODEL_INDEX.md` or use `pyimgano-benchmark --list-models`.
