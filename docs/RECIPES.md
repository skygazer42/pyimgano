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

### `industrial-adapt-highres`

High-resolution tiling preset for industrial inspection images (2K/4K).

This recipe sets practical defaults when the config does not explicitly request tiling:

- `adaptation.tiling.tile_size = 512`
- `adaptation.tiling.stride = 384` (overlap to reduce seam artifacts)
- `adaptation.tiling.map_reduce = "hann"` (seam-reducing blending)
- `adaptation.postprocess` default: percentile normalization + light gaussian blur
- `adaptation.save_maps = true` (so outputs can be audited and reused for inference)

Use this when you know your images are too large for a single `Resize(256,256)` style flow.

### `industrial-adapt-fp40`

False-positive reduction preset intended for the deploy flow:

1) run workbench and export `infer_config.json`
2) run inference with `pyimgano-infer --infer-config ... --defects`

This recipe forces `defects.enabled=true` and applies best-effort FP40 defaults for:

- ROI gating (`defects.roi_xyxy_norm`)
- border suppression (`defects.border_ignore_px`)
- anomaly-map smoothing + hysteresis
- shape filters + small-area removal
- region merging + stable `max_regions`

It also sets `adaptation.save_maps=true` so that infer-config can auto-enable maps/postprocess.

### `micro-finetune-autoencoder`

Micro-finetune recipe intended for small autoencoder-style models. It writes a
checkpoint under `checkpoints/` and produces a workbench-style run report.

For end-to-end “train + eval” runs, you can also enable the `training` section
inside `industrial-adapt`.

### `anomalib-train` (optional; placeholder)

Recipe skeleton intended for future end-to-end anomalib training integration.

For now:

- train in anomalib
- use `vision_*_anomalib` backends or `vision_anomalib_checkpoint` for evaluation/inference in `pyimgano`

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

### Defects export knobs (mask + regions + ROI)

The `defects` block is stored in the workbench config and exported into
`artifacts/infer_config.json` when you run `pyimgano-train --export-infer-config`.
This is intended for deployment-style inference where you want the same settings
to travel with the model + threshold.

Example:

```json
{
  "defects": {
    "enabled": true,
    "pixel_threshold": 0.5,
    "pixel_threshold_strategy": "fixed",
    "pixel_normal_quantile": 0.999,
    "mask_format": "png",
	    "roi_xyxy_norm": [0.1, 0.1, 0.9, 0.9],
	    "min_area": 0,
	    "min_score_max": null,
	    "min_score_mean": null,
	    "open_ksize": 0,
	    "close_ksize": 0,
	    "fill_holes": false,
	    "max_regions": null
	  }
	}
```

Notes:

- Workbench training/eval does not currently run defects extraction; this block is for inference export.
- When you run `pyimgano-infer --infer-config ... --defects`, `pyimgano-infer` uses the exported `defects` block
  as **defaults** for defects extraction (pixel threshold, ROI, morphology, min-area, mask format, etc.).
  Explicit CLI flags always override these defaults.
- See:
  - `docs/INDUSTRIAL_INFERENCE.md` (defects export overview)
  - `docs/CLI_REFERENCE.md` (`pyimgano-infer` flags)
  - `examples/configs/industrial_adapt_defects_roi.json` (template)

### Training knobs (micro-finetune + checkpoints)

Enable micro-finetune (best-effort) and persist a checkpoint:

```json
{
  "training": {
    "enabled": true,
    "epochs": 5,
    "lr": 0.001,
    "checkpoint_name": "model.pt"
  }
}
```

Notes:

- Checkpoints are written under `checkpoints/<category>/<checkpoint_name>`.
- `epochs` / `lr` are passed to `detector.fit(...)` when the model supports them.

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

## Inference from a workbench run

`pyimgano-infer` can reuse a workbench run directory:

```bash
pyimgano-infer --from-run runs/my_run --input /path/to/images --save-jsonl out.jsonl
```

If the run contains multiple categories, pass:

```bash
pyimgano-infer --from-run runs/my_run --from-run-category bottle --input /path/to/images
```

Best-effort behavior:

- loads model settings from `config.json`
- loads a checkpoint when present in `report.json` (when the detector supports it)
- sets `detector.threshold_` from `report.json` when available

---

## Model weights policy

`pyimgano` does **not** ship model weights inside the wheel.

For models that download weights (e.g. torchvision / OpenCLIP), weights are
cached on disk by their upstream libraries (commonly under `~/.cache`).

Notes:

- Workbench run artifacts (`runs/.../`) are intended to stay lightweight (reports, JSONL, `.npy` maps, and small checkpoints).
- To relocate caches on servers/containers, set environment variables such as:
  - `TORCH_HOME`
  - `HF_HOME` / `TRANSFORMERS_CACHE`
  - `XDG_CACHE_HOME`
