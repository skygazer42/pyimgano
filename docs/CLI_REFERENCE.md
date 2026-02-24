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
- `--from-run RUN_DIR` — load model/threshold/checkpoint from a prior `pyimgano-train` workbench run
  - If the run contains multiple categories, pass `--from-run-category NAME`.
- `--infer-config PATH` — load model/threshold/checkpoint from an exported workbench infer-config
  - For example: `runs/.../artifacts/infer_config.json`
  - If the infer-config contains multiple categories, pass `--infer-category NAME`.

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
