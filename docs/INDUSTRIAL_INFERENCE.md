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

Each JSONL line includes:

- `score` (float)
- `label` (0/1 if `threshold_` is available)
- `input` (path)
- `anomaly_map.path` + `shape` + `dtype` (if map exported)
