# Feature Extractors

PyImgAno classical (feature-based) detectors consume a 2D feature matrix of shape
`(n_samples, n_features)`. For vision workflows, the model wrappers accept a
`feature_extractor` object that implements:

- `extract(inputs) -> np.ndarray`
- Optional: `fit(inputs, y=None) -> self`

This lets you plug in custom feature extractors or use the built-in extractor
registry.

## CLI Discovery

List available extractors:

```bash
pyimgano-benchmark --list-feature-extractors
pyimgano-benchmark --list-feature-extractors --json
pyimgano-benchmark --list-feature-extractors --feature-tags texture
```

Show info for one extractor:

```bash
pyimgano-benchmark --feature-info hog
pyimgano-benchmark --feature-info hog --json
```

## Python Usage

Create an extractor by name:

```python
from pyimgano.features import create_feature_extractor

hog = create_feature_extractor("hog", resize_hw=(128, 128))
X = hog.extract([img0, img1])  # -> (2, n_features)
```

Use a JSON-friendly spec (useful for config files):

```python
from pyimgano.features.registry import resolve_feature_extractor

spec = {"name": "lbp", "kwargs": {"n_points": 8, "radius": 1.0, "method": "uniform"}}
lbp = resolve_feature_extractor(spec)
```

Use a spec directly in model kwargs:

```python
from pyimgano.models import create_model

det = create_model(
    "vision_ecod",
    contamination=0.1,
    feature_extractor={"name": "hog", "kwargs": {"resize_hw": [64, 64]}},
)
```

## Built-in Extractors

Names are registered under `pyimgano.features.list_feature_extractors()`.

| Name | Tags | Description |
|---|---|---|
| `identity` | `embeddings` | No-op: inputs are already feature vectors |
| `hog` | `texture` | Histogram of Oriented Gradients |
| `lbp` | `texture` | Local Binary Pattern histogram |
| `gabor_bank` | `texture` | Gabor filter bank response stats |
| `color_hist` | `color` | Per-channel color histograms (HSV/LAB/RGB/BGR) |
| `edge_stats` | `edges` | Canny + Sobel statistics |
| `fft_lowfreq` | `frequency` | Low-frequency FFT energy ratios |
| `patch_stats` | `statistics` | Patch-grid stats (mean/std/skew/kurt) |
| `multi` | `pipeline` | Concatenate multiple extractors |
| `pca_projector` | `pipeline` | PCA projection on top of a base extractor |
| `standard_scaler` | `pipeline` | StandardScaler on top of a base extractor |

