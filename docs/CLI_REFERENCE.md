# CLI Reference

PyImgAno provides three primary CLIs:

- `pyimgano-benchmark` — one-click industrial benchmarking + run artifacts
- `pyimgano-infer` — JSONL inference over images/videos (path-driven)
- `pyimgano-robust-benchmark` — robustness evaluation (clean + corruptions)

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

- `--calibration-quantile Q` — override score threshold quantile

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
  --calibration-quantile 0.995 \
  --save-jsonl out.jsonl
```

Optional:

- `--include-maps` + `--save-maps DIR` — write anomaly maps as `.npy`

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

## Notes

- Many models have optional backends; when required dependencies are missing, error messages include install hints.
- For a full model catalog, see `docs/MODEL_INDEX.md` or use `pyimgano-benchmark --list-models`.

