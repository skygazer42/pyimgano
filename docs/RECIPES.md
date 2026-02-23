# Recipes (Industrial Workbench)

`pyimgano` includes a **recipe-driven workbench layer** for industrial anomaly
detection workflows.

The goal is to make “the industrial loop” reproducible and comparable:

1) dataset alignment (train/test split, resize, input mode)
2) adaptation knobs (tiling, anomaly-map postprocess, threshold calibration)
3) evaluation (image-level + optional pixel-level metrics)
4) artifacts (reports + per-image JSONL + optional anomaly maps)

The primary entrypoint is:

```bash
pyimgano-train --config cfg.json
```

See `docs/CLI_REFERENCE.md` for CLI flags.

---

## Recipe discovery

```bash
pyimgano-train --list-recipes
pyimgano-train --recipe-info industrial-adapt
```

Add `--json` to emit machine-readable outputs.

---

## Builtin recipes

### `industrial-adapt`

An **adaptation-first** benchmark recipe intended for industrial inspection.

It runs a standardized loop:

- load dataset split (paths or numpy inputs)
- create detector from the model registry
- optional high-resolution tiling wrapper (for 2K/4K images)
- fit on normal samples
- calibrate a score threshold from train samples
- infer on test samples (optionally extracting anomaly maps)
- optional anomaly-map postprocess (normalize / blur / morphology)
- evaluate metrics and write artifacts

---

## Config format

Config files are **JSON by default** (`.json`).

YAML (`.yml` / `.yaml`) is supported only when `PyYAML` is installed:

```bash
pip install PyYAML
```

### Minimal config

```json
{
  "recipe": "industrial-adapt",
  "seed": 123,
  "dataset": {
    "name": "mvtec",
    "root": "/path/to/mvtec_ad",
    "category": "bottle",
    "resize": [256, 256],
    "input_mode": "paths"
  },
  "model": {
    "name": "vision_patchcore",
    "device": "cuda",
    "preset": "industrial-balanced",
    "pretrained": true,
    "contamination": 0.1
  },
  "output": {
    "output_dir": "runs/my_run",
    "save_run": true,
    "per_image_jsonl": true
  }
}
```

### Adaptation knobs (tiling + maps + postprocess)

```json
{
  "adaptation": {
    "save_maps": true,
    "tiling": {
      "tile_size": 512,
      "stride": 384,
      "score_reduce": "max",
      "score_topk": 0.1,
      "map_reduce": "hann"
    },
    "postprocess": {
      "normalize": true,
      "normalize_method": "percentile",
      "percentile_range": [1.0, 99.0],
      "gaussian_sigma": 1.0,
      "morph_open_ksize": 3,
      "morph_close_ksize": 3
    }
  }
}
```

Notes:

- `adaptation.save_maps=true` writes `.npy` anomaly maps under `artifacts/maps/` and
  annotates `per_image.jsonl` records with map metadata (`path`, `shape`, `dtype`).
- `adaptation.tiling.tile_size` enables tiled inference for high-resolution images.
- `adaptation.postprocess` is applied to anomaly maps (when extracted).

---

## Artifact layout

When `output.save_run=true`, `pyimgano-train` writes:

```
<run_dir>/
  report.json
  config.json
  environment.json
  categories/<cat>/report.json
  categories/<cat>/per_image.jsonl
  checkpoints/...
  artifacts/maps/... (optional)
```

All `report.json` payloads include:

- `schema_version`
- `timestamp_utc`
- `pyimgano_version`

---

## Model weights policy

`pyimgano` does **not** ship model weights inside the wheel.

For models that download weights (e.g. torchvision / OpenCLIP), weights are
cached on disk by their upstream libraries (commonly under `~/.cache`).

