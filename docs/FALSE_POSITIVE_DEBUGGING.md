# False Positive Debugging (Industrial Inspection / Defect Masks)

This guide is for the practical situation where your model **detects anomalies**, but you get too many
false positives in production:

- defect masks light up on borders / fixtures / background
- small speckles trigger regions
- tiling seams become “defects”
- lighting drift shifts the anomaly map distribution

The goal is **not** to claim a single “best” setting, but to give you a stable, reproducible debugging loop.

---

## Quick checklist (90% of FP issues)

1) **ROI gate first**: if only part of the frame is inspectable, set ROI (don’t fight background FP with thresholds).
2) **Border suppress**: many industrial cameras have border artifacts; enable `border_ignore_px`.
3) **Enable overlays**: visualize anomaly heatmap + mask + regions before tuning.
4) **Add smoothing + hysteresis**: remove salt/pepper and keep only connected-to-seed pixels.
5) **Shape + area filters**: remove thin strips and tiny blobs.
6) **Check tiling overlap / blending**: seams often look like defects when stride is too large.
7) **Address illumination drift**: apply a consistent illumination/contrast normalization chain.

---

## Recommended debugging command

Start from a workbench export (preferred):

```bash
pyimgano-infer \
  --infer-config /path/to/run_dir/artifacts/infer_config.json \
  --input /path/to/images \
  --defects \
  --save-jsonl /tmp/results.jsonl \
  --save-overlays /tmp/overlays \
  --save-masks /tmp/masks \
  --mask-format png
```

Notes:

- `--save-overlays` is the fastest way to understand **why** you’re getting false positives.
- If your run is large, add `--batch-size 8` and stream outputs (JSONL is written incrementally).

---

## Step-by-step tuning (in order)

### 1) ROI gating

If your camera view contains a fixture, background, or conveyor belt, use ROI gating so those pixels
do not participate in defect extraction:

```bash
--roi-xyxy-norm 0.1 0.1 0.9 0.9
```

Practical notes:

- ROI affects **defects output** (mask/regions) by default, not the image-level score.
- If you calibrate pixel threshold using `normal_pixel_quantile`, ROI also limits which pixels are used.

### 2) Border suppression

Suppress a border band (in anomaly-map pixel space):

```bash
--defect-border-ignore-px 2
```

This is useful for:

- sensor borders
- padding artifacts
- tiling seam “hot edges”

### 3) Smooth anomaly map before thresholding

Start with median smoothing:

```bash
--defect-map-smoothing median \
--defect-map-smoothing-ksize 3
```

If you have larger “sparkle” noise, increase to `5` (but note it can blur small defects).

### 4) Use hysteresis thresholding

Hysteresis reduces speckle by requiring low-confidence pixels to be connected to a high-confidence seed:

```bash
--defect-hysteresis
```

Optionally set explicit thresholds:

```bash
--defect-hysteresis-low 0.45 \
--defect-hysteresis-high 0.60
```

### 5) Filter regions by shape + area

These are the most common “production-grade” post-filters:

```bash
--defect-min-area 16 \
--defect-min-fill-ratio 0.15 \
--defect-max-aspect-ratio 6.0 \
--defect-min-solidity 0.8
```

Interpretation:

- `min_area`: drop tiny blobs that are usually noise.
- `min_fill_ratio`: reject “stringy” regions that barely fill their bbox.
- `max_aspect_ratio`: reject long thin strips (common in seams and borders).
- `min_solidity`: reject jagged / hollow shapes.

### 6) Threshold on per-region confidence

If masks are still too noisy, add a per-region score gate:

```bash
--defect-min-score-max 0.6
```

This discards regions whose **max** anomaly-map value is too low.

### 7) Merge nearby fragments (regions only)

If one physical defect becomes multiple small regions, merge them in the regions list:

```bash
--defect-merge-nearby \
--defect-merge-nearby-max-gap-px 1
```

This keeps the binary mask unchanged (so you can still do pixel counting), but yields a cleaner regions list.

### 8) Fix tiling seams (high-resolution images)

If you use tiling, overlap matters:

```bash
--tile-size 512 \
--tile-stride 384 \
--tile-map-reduce hann
```

Guideline:

- use overlap (`stride < tile_size`)
- use a weighted blender (`hann`/`gaussian`) rather than `max` for seam stability

### 9) Control illumination drift (preprocessing)

Lighting drift is a major FP driver in industrial inspection.

Use the opt-in, uint8-preserving preset chain:

```python
import cv2
from pyimgano.preprocessing import IlluminationContrastKnobs, apply_illumination_contrast

img = cv2.imread("frame.png")
knobs = IlluminationContrastKnobs(
    white_balance="gray_world",
    homomorphic=True,
    clahe=True,
    gamma=0.9,
)
img2 = apply_illumination_contrast(img, knobs=knobs)
```

---

## Auditability / deployment tips

- Validate an exported infer-config before deploying:

```bash
pyimgano-validate-infer-config /path/to/infer_config.json
```

- Export a deploy bundle containing `infer_config.json` + referenced checkpoints:

```bash
pyimgano-train --config cfg.json --export-deploy-bundle
```

This produces `<run_dir>/deploy_bundle/` for copying to servers/containers.
