# CLI Reference

PyImgAno provides the following CLIs:

- `pyimgano-benchmark` — one-click industrial benchmarking + run artifacts
- `pyimgano-demo` — minimal offline demo (creates a tiny custom dataset + runs a suite/sweep)
- `pyimgano-train` — recipe-driven workbench runs (adaptation-first; optional micro-finetune)
- `pyimgano-infer` — JSONL inference over images/videos (path-driven)
- `pyimgano-defects` — standalone anomaly-map → mask → regions defects export
- `pyimgano-robust-benchmark` — robustness evaluation (clean + corruptions)
- `pyimgano-doctor` — environment + optional dependency (extras) sanity check
- `pyimgano-manifest` — generate a JSONL manifest from a `custom`-layout dataset tree
- `pyimgano-datasets` — dataset converter discovery + metadata
- `pyimgano-synthesize` — anomaly synthesis + manifest generation
- `pyimgano-validate-infer-config` — validate an exported `infer_config.json` before deployment
- `pyimgano-features` — feature/embedding extractor discovery + extraction utilities
- `pyimgano-export-torchscript` — export a torchvision backbone to TorchScript (offline-safe by default)
- `pyimgano-export-onnx` — export a torchvision backbone to ONNX (offline-safe by default)

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
  --pretrained \
  --device cuda
```

Notes:
- CLIs default to **offline-safe** behavior (`--no-pretrained`). Use `--pretrained` explicitly when you want upstream weights (may download).
- `--model` can be either a registered model name (e.g. `vision_patchcore`) or a **model preset name**
  (e.g. `industrial-pixel-mad-map`). To discover presets, use `pyimgano-infer --list-model-presets`.

### Discovery

- List models: `pyimgano-benchmark --list-models`
- Load third-party plugins (entry points) before discovery: `pyimgano-benchmark --plugins --list-models`
- Filter models by tags: `pyimgano-benchmark --list-models --tags vision,deep`
- Model info (constructor signature + accepted kwargs): `pyimgano-benchmark --model-info vision_patchcore`
- List dataset categories: `pyimgano-benchmark --list-categories --dataset mvtec --root /path/to/mvtec_ad`
- List manifest categories: `pyimgano-benchmark --list-categories --dataset manifest --manifest-path /path/to/manifest.jsonl`
- List curated industrial baseline suites: `pyimgano-benchmark --list-suites`
- Suite contents (resolved baselines): `pyimgano-benchmark --suite-info industrial-v1`
- List curated suite sweep profiles (small grid searches): `pyimgano-benchmark --list-sweeps`
- Sweep contents (variants + overrides): `pyimgano-benchmark --sweep-info industrial-small`

### Baseline Suites (Industrial)

Suites are curated packs of **multiple model presets** intended for industrial algorithm selection.

Built-in suites: `industrial-ci`, `industrial-v1`, `industrial-v2`, `industrial-v3`, `industrial-v4` (use `--list-suites` for the full list).

Example:

```bash
pyimgano-benchmark \
  --dataset mvtec \
  --root /path/to/mvtec_ad \
  --category bottle \
  --suite industrial-v1 \
  --device cpu \
  --no-pretrained
```

Flags:

- `--suite NAME` — run a curated suite (instead of a single `--model`)
- `--suite-max-models N` — limit number of baselines (smoke/debug)
- `--suite-include NAME[,NAME]` — run only selected suite baselines (comma-separated or repeatable)
- `--suite-exclude NAME[,NAME]` — skip selected suite baselines (comma-separated or repeatable)
- `--suite-continue-on-error/--no-suite-continue-on-error` — keep running when a baseline errors or has missing optional deps
- `--suite-export csv|md|both` — write `leaderboard.*`, `best_by_baseline.*`, and `skipped.*` tables into the suite output directory (requires `--save-run`)
- `--suite-export-best-metric NAME` — metric used for `best_by_baseline.*` tables (default: `auroc`). Pixel metrics require `--pixel`.
- `--suite-sweep SPEC` — run a small parameter sweep (grid search) per baseline and rank variants.
  `SPEC` can be a built-in sweep name (discover with `--list-sweeps`), a JSON file path, or inline JSON.
- `--suite-sweep-max-variants N` — cap the number of sweep variants per baseline (excluding base). Example: `--suite-sweep-max-variants 1`

Suite artifacts (when `--save-run` is enabled):

- `<suite_dir>/report.json` — aggregated suite report (ranking + skipped baselines)
- `<suite_dir>/config.json`, `<suite_dir>/environment.json`
- `<suite_dir>/leaderboard.csv`, `<suite_dir>/skipped.csv` (when `--suite-export csv|both`)
- `<suite_dir>/best_by_baseline.csv` (when `--suite-export csv|both`, best variant per baseline by AUROC; most useful with `--suite-sweep`)
- `<suite_dir>/leaderboard.md`, `<suite_dir>/skipped.md` (when `--suite-export md|both`)
- `<suite_dir>/best_by_baseline.md` (when `--suite-export md|both`, best variant per baseline by AUROC; most useful with `--suite-sweep`)
- `<suite_dir>/models/<baseline_name>/...` — per-baseline run artifacts
- `<suite_dir>/models/<baseline_name>/variants/<variant>/...` (when `--suite-sweep` is enabled)

Optional extras:

- Some suite entries are marked optional and are **skipped** when extras are not installed.
  Skip reasons include actionable install hints like `pip install 'pyimgano[skimage]'` or `pip install 'pyimgano[torch]'`.

#### Custom sweep JSON

You can pass a JSON sweep plan file to `--suite-sweep` (or prefix the path with `@`):

```json
{
  "name": "my-sweep",
  "description": "Tiny sweep for NCC window sizes",
  "variants_by_entry": {
    "industrial-template-ncc-map": [
      {"name": "win_7", "override": {"window_hw": [7, 7]}},
      {"name": "win_21", "override": {"window_hw": [21, 21]}}
    ]
  }
}
```

Run it:

```bash
pyimgano-benchmark \
  --dataset mvtec \
  --root /path/to/mvtec_ad \
  --category bottle \
  --suite industrial-ci \
  --suite-sweep ./my_sweep.json \
  --no-pretrained
```

### Manifest dataset

Benchmark a manifest JSONL (paths mode):

```bash
pyimgano-benchmark \
  --dataset manifest \
  --root /path/to/fallback_root \
  --manifest-path /path/to/manifest.jsonl \
  --category bottle \
  --model vision_patchcore \
  --pretrained \
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

### Model Presets (Shortcuts)

Presets are just **named (model + kwargs)** pairs that keep industrial command lines short while staying reproducible.

- List presets: `pyimgano-infer --list-model-presets`
- Show preset details (model/kwargs/description): `pyimgano-infer --model-preset-info industrial-pixel-mad-map`

Example:

```bash
pyimgano-infer \
  --model-preset industrial-pixel-mad-map \
  --train-dir /path/to/train/normal \
  --input /path/to/inputs \
  --include-maps \
  --save-maps /tmp/pyimgano_maps \
  --save-jsonl out.jsonl
```

### Deployment-Friendly Embeddings

If you want the “embedding + core” industrial route without relying on upstream model registries at inference time,
use one of the deployment wrapper models and pass `--checkpoint-path`:

- TorchScript: `vision_torchscript_ecod`, `vision_torchscript_knn_cosine_calibrated`, ...
- ONNX Runtime: `vision_onnx_ecod`, `vision_onnx_knn_cosine_calibrated`, ...

Example (ONNX embeddings + ECOD):

```bash
pyimgano-infer \
  --model vision_onnx_ecod \
  --checkpoint-path /path/to/resnet18_embed.onnx \
  --train-dir /path/to/train/normal \
  --input /path/to/inputs \
  --save-jsonl out.jsonl
```

Optional:

- `--include-maps` + `--save-maps DIR` — write anomaly maps as `.npy`
- High-resolution tiling (optional; for 2K/4K inspection images):
  - `--tile-size N` — run tiled inference (wraps the detector in `TiledDetector`)
  - `--tile-stride N` — tile overlap stride (default: tile-size; smaller = more overlap = fewer seams)
  - `--tile-map-reduce max|mean|hann|gaussian` — blend overlapping tile maps (`hann`/`gaussian` reduce seams)
  - `--tile-score-reduce max|mean|topk_mean` + `--tile-score-topk` — aggregate tile scores into an image score
- `--seed INT` — best-effort deterministic seeding (also passed as `random_seed/random_state` when supported)
- `--batch-size N` — run inference in chunks (preserves output order; can reduce peak memory)
- `--profile` — print stage timing summary to stderr (load model, fit/calibrate, infer, artifacts)
- `--amp` — best-effort AMP/autocast for torch-backed models (requires torch + CUDA; otherwise runs without AMP)
- `--include-anomaly-map-values` — embed raw anomaly-map values in JSONL (debug only; very large output)
- `--defects` — export industrial defect structures (binary mask + connected-component regions)
  - `--defects-preset industrial-defects-fp40` — FP reduction defaults (ROI/border/smoothing/hysteresis/shape filters)
  - `--defects-regions-jsonl PATH` — write per-image regions payloads to a dedicated JSONL file
  - `--save-masks DIR` + `--mask-format png|npy|npz` (`npz` is compressed numpy; good for large batches)
  - `--defects-mask-space roi|full` — when ROI is set, export ROI-only or full-size masks (regions are always ROI-gated)
  - `--defects-mask-dilate INT` — optional mask dilation for industrial fill/coverage
  - `--save-overlays DIR` — save per-image debugging overlays (original + heatmap + mask outline/fill)
  - `--defects-image-space` — add `bbox_xyxy_image` to regions when image size is available
  - Pixel threshold options:
    - `--pixel-threshold FLOAT` + `--pixel-threshold-strategy fixed`
    - `--pixel-threshold-strategy infer_config` (uses `defects.pixel_threshold` from `infer_config.json` / a workbench run)
    - `--pixel-threshold-strategy normal_pixel_quantile` (requires `--train-dir`; uses `--pixel-normal-quantile`)
      - If selected and `--train-dir` is provided, `pyimgano-infer` recalibrates from normal/train maps even if `infer_config.json` contains `defects.pixel_threshold`.
  - When running with `--infer-config` or `--from-run`, the exported `defects.*` settings are used as defaults
    (ROI, morphology, min-area, mask format, max regions, pixel threshold strategy/quantile, etc.). CLI flags override.
  - When running with `--infer-config` or `--from-run`, exported preprocessing defaults (e.g. `preprocessing.illumination_contrast`)
    are applied automatically for deploy consistency (when present).
    - Note: `preprocessing.illumination_contrast` requires a numpy-capable model (tag: `numpy`), otherwise you’ll see
      `PREPROCESSING_REQUIRES_NUMPY_MODEL`.
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
  - Mask morphology (optional):
    - `--defect-open-ksize INT` / `--defect-close-ksize INT`
    - `--defect-fill-holes`
- `--from-run RUN_DIR` — load model/threshold/checkpoint from a prior `pyimgano-train` workbench run
  - If the run contains multiple categories, pass `--from-run-category NAME`.
- `--infer-config PATH` — load model/threshold/checkpoint from an exported workbench infer-config
  - For example: `runs/.../artifacts/infer_config.json`
  - If the infer-config contains multiple categories, pass `--infer-category NAME`.

Defects export example:

```bash
pyimgano-infer \
  --model vision_patchcore \
  --pretrained \
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

## `pyimgano-validate-infer-config`

Validates an exported `infer_config.json` from the workbench before deployment.

```bash
pyimgano-validate-infer-config runs/.../artifacts/infer_config.json
```

Notes:

- For multi-category infer-configs, pass `--infer-category NAME`.
- To skip file existence checks (portable configs), pass `--no-check-files`.
- To print the normalized payload, pass `--json`.

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

Common outputs:

- `--export-infer-config` — write `artifacts/infer_config.json` into the run directory
- `--export-deploy-bundle` — write `deploy_bundle/` (includes `infer_config.json` + referenced checkpoints)
  - Validate the bundle with: `pyimgano-validate-infer-config deploy_bundle/infer_config.json`

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
  --pretrained \
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

---

## `pyimgano-export-torchscript`

Export a torchvision backbone (classification head stripped) as a TorchScript `.pt` file.

Requires:

- `pip install "pyimgano[torch]"`

Example (offline-safe):

```bash
pyimgano-export-torchscript \
  --backbone resnet18 \
  --out /tmp/resnet18_backbone.pt
```

Notes:

- `--pretrained` is **off by default** to avoid implicit weight downloads.
- Use `--method trace|script` depending on your deployment constraints (default: `trace`).

---

## `pyimgano-export-onnx`

Export a torchvision backbone (classification head stripped) as an ONNX `.onnx` file.

Requires:

- `pip install "pyimgano[torch]"` (export)
- `pip install "pyimgano[onnx]"` (recommended; needed for `--verify`, and required by newer `torch.onnx.export` flows via `onnxscript`)

Example (offline-safe):

```bash
pyimgano-export-onnx \
  --backbone resnet18 \
  --out /tmp/resnet18_backbone.onnx
```

Notes:

- `--pretrained` is **off by default** to avoid implicit weight downloads.
- By default, export uses `--dynamic-batch` (deploy-friendly).
- `--verify` (default: true) checks that the exported file loads in `onnx` and `onnxruntime`.

## Notes

- Many models have optional backends; when required dependencies are missing, error messages include install hints.
- For a full model catalog, see `docs/MODEL_INDEX.md` or use `pyimgano-benchmark --list-models`.
