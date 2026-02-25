# Industrial Inference (numpy-first)

This guide documents the **numpy-first** inference helpers in `pyimgano`.

The goal is to make production integration (video pipelines, PLC systems, backend services)
less error-prone by:

- requiring an explicit `ImageFormat` (no “auto guessing”)
- canonicalizing images to `RGB / uint8 / HWC`
- returning structured inference results (score, label, optional anomaly map)

## 1) Image formats (explicit)

In production you often have frames already decoded in memory, usually as:

- OpenCV frames: **BGR / uint8 / HWC**
- Deep learning tensors: **RGB / float32 / CHW** in `[0, 1]`

Use `ImageFormat` + `normalize_numpy_image` to convert into the canonical format:

```python
import numpy as np

from pyimgano.inputs import ImageFormat, normalize_numpy_image

bgr_u8_hwc: np.ndarray = ...
rgb_u8_hwc = normalize_numpy_image(bgr_u8_hwc, input_format=ImageFormat.BGR_U8_HWC)
```

## 2) Inference API

The `pyimgano.inference` API is detector-agnostic: it calls `decision_function`
and (optionally) `get_anomaly_map` / `predict_anomaly_map` if available.

```python
import numpy as np

from pyimgano.inference import calibrate_threshold, infer
from pyimgano.inputs import ImageFormat
from pyimgano.models import create_model
from pyimgano.postprocess.anomaly_map import AnomalyMapPostprocess

# A small, no-download demo config (pretrained=False avoids weight downloads).
detector = create_model(
    "vision_padim",
    pretrained=False,
    device="cpu",
    image_size=64,
    d_reduced=8,
    projection_fit_samples=1,
    covariance_eps=0.1,
)

# Suppose you have OpenCV frames (BGR/u8/HWC) already in memory:
train_frames_bgr = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(8)]
test_frames_bgr = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(2)]

# Most detectors still need a `fit()` step.
# For training, pass canonical RGB/u8/HWC (normalize first if you have BGR frames).
from pyimgano.inputs import normalize_numpy_image

train_frames_rgb = [
    normalize_numpy_image(frame, input_format=ImageFormat.BGR_U8_HWC) for frame in train_frames_bgr
]
detector.fit(train_frames_rgb)

# Optional: set a stricter threshold based on normal calibration frames.
calibrate_threshold(detector, train_frames_bgr, input_format=ImageFormat.BGR_U8_HWC, quantile=0.995)

post = AnomalyMapPostprocess(normalize=True, normalize_method="minmax")
results = infer(
    detector,
    test_frames_bgr,
    input_format=ImageFormat.BGR_U8_HWC,
    include_maps=True,
    postprocess=post,
)

for r in results:
    print(r.score, r.label, None if r.anomaly_map is None else r.anomaly_map.shape)
```

Notes on threshold calibration:

- A common industrial heuristic is `quantile = 1 - contamination` (e.g. contamination=0.1 → quantile=0.9).
- For stricter false-positive control on “clean” normal sets, `0.995` or `0.999` are common starting points.

## 3) Capability tags (`numpy`, `pixel_map`)

Many detectors accept different input types and expose different outputs.
Use CLI discovery with capability tags:

```bash
pyimgano-benchmark --list-models --tags numpy
pyimgano-benchmark --list-models --tags numpy,pixel_map
```

## 4) `pyimgano-infer` CLI (JSONL + optional map export)

For service integration, `pyimgano-infer` provides a small JSONL-based CLI:

```bash
pyimgano-infer \
  --model vision_patchcore \
  --train-dir /path/to/normal/images \
  --input /path/to/test/images_or_file \
  --include-maps \
  --save-maps /tmp/pyimgano_maps \
  --save-jsonl /tmp/pyimgano_results.jsonl
```

By default, `pyimgano-infer` auto-calibrates `threshold_` from train scores when `--train-dir`
is provided and the detector doesn’t already set `threshold_`. The default quantile matches
`pyimgano-benchmark`: `1 - contamination` when available, else `0.995`.

Each JSONL line includes:

- `score` (float)
- `label` (0/1 if `threshold_` is available)
- `input` (path)
- `anomaly_map.path` + `shape` + `dtype` (if map exported)

## 5) High-resolution tiling (2K/4K inspection images)

Many industrial inspection images are too large to score reliably after a single
`Resize(224,224)` style preprocessing. For detectors that support numpy inputs + pixel maps,
wrap them with `TiledDetector` to run overlapping-window inference.

Python API:

```python
from pyimgano.inference.tiling import TiledDetector

tiled = TiledDetector(
    detector=detector,
    tile_size=512,
    stride=384,              # overlap improves seam quality
    score_reduce="topk_mean",
    score_topk=0.1,
    # Map blending:
    # - "max": sharp but can leave seams
    # - "mean": smooth but can blur peaks
    # - "hann"/"gaussian": weighted blending to reduce seams
    map_reduce="hann",
)
```

CLI flags (JSONL + tiling):

```bash
pyimgano-infer \
  --model vision_patchcore \
  --train-dir /path/to/normal/images \
  --input /path/to/test/images_or_file \
  --include-maps \
  --tile-size 512 \
  --tile-stride 384 \
  --tile-map-reduce hann
```

## 6) Defects export (mask + regions + ROI)

Industrial deployments often need more than a heatmap:

- a **binary defect mask** (for overlay / QC rules)
- **regions/instances** (bbox / area / centroid) for downstream analytics
- optional **ROI** gating so fixtures/background do not dominate false positives

`pyimgano-infer` can emit this structure as a `defects` block in JSONL when enabled:

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
  --save-jsonl /tmp/pyimgano_results.jsonl
```

If you exported an `infer_config.json` from the workbench (`pyimgano-train --export-infer-config`),
you can reuse it for inference:

```bash
pyimgano-infer \
  --infer-config /path/to/run_dir/artifacts/infer_config.json \
  --input /path/to/inputs \
  --defects \
  --save-masks /tmp/pyimgano_masks \
  --save-jsonl /tmp/pyimgano_results.jsonl
```

If the infer-config contains `defects.pixel_threshold`, you can omit `--pixel-threshold`.

If the infer-config contains other `defects.*` settings (ROI, morphology, min-area, mask format, etc.),
`pyimgano-infer` uses them as defaults when `--defects` is enabled (CLI flags override).

Notes:

- `--defects` implies `--include-maps` (defects are derived from anomaly maps).
- ROI gating affects **defects output only** by default (mask/regions), not image-level `score`/`label`.
- When calibrating pixel threshold via `normal_pixel_quantile` and ROI is set, calibration uses ROI pixels only.
- Use `--defect-border-ignore-px N` to suppress common edge artifacts (padding/tiling seams/sensor borders).
- Use `--defect-map-smoothing median|gaussian|box` to reduce single-pixel speckles before thresholding.
- Use `--defect-hysteresis` to keep low anomaly pixels connected to high-confidence seeds (reduces speckle).
- Use shape filters (`--defect-min-fill-ratio`, `--defect-max-aspect-ratio`, `--defect-min-solidity`) to remove “long thin” border strips and non-compact fragments.
- Use `--defect-min-score-max` / `--defect-min-score-mean` to drop low-confidence speckles after thresholding.
- Use `--defect-merge-nearby --defect-merge-nearby-max-gap-px N` to merge nearby fragments in the **regions list** (mask remains unchanged).
- Use `--defects-image-space` to add `bbox_xyxy_image` for downstream systems that work in original image coordinates.
- Use `--save-overlays DIR` to export per-image FP debugging overlays (original + heatmap + mask outline/fill).
- Defect coordinates (`bbox_xyxy`, `centroid_xy`) are in **anomaly-map pixel space**.
- Pixel threshold provenance is always emitted as `defects.pixel_threshold_provenance` for auditability.
