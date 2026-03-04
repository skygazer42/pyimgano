# Recipes: Pixel-First Baselines (SSIM / NCC / Phase-Corr)

This page contains copy‑pastable recipes for **pixel-first** industrial anomaly detection.

Pixel-first means:

- the detector produces an **anomaly map** (per-pixel score, higher = more anomalous)
- maps can be exported directly (QA/debug), or turned into **defects** (mask + connected components)

This route is often the fastest industrial on-ramp when you have **aligned template-style inspection**
(screens, labels, PCBs, structured surfaces).

## When to use

Good fit:

- “golden reference” / template inspection (normal images are very similar)
- controlled station, stable camera, stable pose
- you want a deploy-friendly output: defect mask + regions

Potentially poor fit:

- large pose/viewpoint changes (use tiling + stronger features, or multi-view modeling)
- heavily textured normals (thresholding may need more tuning)

---

## Models

### 1) SSIM template map (`ssim_template_map`)

Fast baseline:

- `fit()` selects 1..K templates from normal images
- inference computes an SSIM map vs the best template
- anomaly map = `1 - ssim_map`

### 2) Structural SSIM map (`ssim_struct_map`)

Like SSIM template map, but computed on edge maps (Canny). Useful when illumination varies
but structure is stable.

### 3) Local NCC template map (`vision_template_ncc_map`)

Local normalized cross-correlation similarity map turned into an anomaly map. Can be robust
for grayscale template inspection when SSIM is too sensitive.

### 4) Phase correlation map (`vision_phase_correlation_map`)

Template-style baseline that is more tolerant to small **global misalignment** than plain
SSIM/NCC.

### 5) Pixel mean abs-diff map (`vision_pixel_mean_absdiff_map`)

Per-pixel statistics baseline for **aligned** inspection:

- `fit()` builds a per-pixel mean template from normal images
- anomaly map is the per-pixel absolute difference to the mean (scaled to `[0,1]`)

Good when your camera/pose is stable and you want a very fast CPU baseline.

### 6) Pixel Gaussian z-score map (`vision_pixel_gaussian_map`)

Like the mean abs-diff baseline, but normalizes per pixel by the learned standard deviation:

- `fit()` learns per-pixel mean + std on normals
- anomaly map is per-pixel z-score `|x - mean| / std`

This is often more robust when illumination/noise differs between runs.

### 7) Pixel robust MAD z-score map (`vision_pixel_mad_map`)

Robust per-pixel baseline for “noisy normal” training data:

- `fit()` learns per-pixel median + MAD (median absolute deviation)
- anomaly map is robust z-score `0.6745 * |x - median| / MAD`

Use this when your normal set may contain a few bad frames/outliers.

---

## Recipe A: Infer maps only (debug/QA)

```bash
pyimgano-infer \
  --model ssim_template_map \
  --train-dir /path/to/train/normal \
  --input /path/to/inputs \
  --include-maps \
  --save-maps /tmp/pyimgano_maps \
  --save-jsonl /tmp/out.jsonl
```

### Preset shortcuts (CLI)

If you prefer JSON-friendly presets over long `--model-kwargs`, these are good starting points:

- `industrial-pixel-gaussian-map` / `industrial-pixel-mad-map` / `industrial-pixel-mean-absdiff-map`
- `industrial-ssim-template-map` / `industrial-phase-correlation-map` / `industrial-template-ncc-map`

Notes:
- `--include-maps` requests maps; some detectors are score-only.
- Use `--postprocess` if you want standard map normalization before exporting.
- All models on this page share the same “map → defects” pipeline. You can swap
  `--model` to any of:
  `vision_pixel_mean_absdiff_map`, `vision_pixel_gaussian_map`, `vision_pixel_mad_map`,
  `vision_template_ncc_map`, `vision_phase_correlation_map`, `ssim_template_map`, etc.

---

## Recipe B: Infer + defects export (mask + regions + overlays)

Use the built-in FP reduction preset as a starting point:

```bash
pyimgano-infer \
  --model ssim_template_map \
  --train-dir /path/to/train/normal \
  --input /path/to/inputs \
  --defects-preset industrial-defects-fp40 \
  --save-masks /tmp/pyimgano_masks \
  --save-overlays /tmp/pyimgano_overlays \
  --defects-regions-jsonl /tmp/defects_regions.jsonl \
  --save-jsonl /tmp/out.jsonl
```

Common tuning knobs:

- Pixel threshold calibration:
  - `--pixel-threshold-strategy normal_pixel_quantile --pixel-normal-quantile 0.999`
- ROI gating (defects only):
  - `--roi-xyxy-norm 0.1 0.1 0.9 0.9`
- Speckle suppression:
  - `--defect-map-smoothing median --defect-map-smoothing-ksize 3`
- Hysteresis (keeps weak pixels connected to strong seeds):
  - `--defect-hysteresis`

---

## Recipe C: High-resolution tiling + defects export (2K/4K)

```bash
pyimgano-infer \
  --model ssim_template_map \
  --train-dir /path/to/train/normal \
  --input /path/to/inputs \
  --tile-size 512 \
  --tile-stride 384 \
  --tile-map-reduce hann \
  --defects-preset industrial-defects-fp40 \
  --save-masks /tmp/pyimgano_masks \
  --save-overlays /tmp/pyimgano_overlays \
  --save-jsonl /tmp/out.jsonl
```

Notes:
- Tiling requires a numpy-capable detector; if your model is paths-only, choose a numpy-tagged model
  or switch to a numpy-first detector.

---

## Tip: Standalone “map → defects” tool

If you already have anomaly maps saved on disk (e.g. from another system), use:

```bash
pyimgano-defects --help
```

This CLI converts anomaly maps into binary masks + connected-component regions with the same
industrial defaults as `pyimgano-infer`.
