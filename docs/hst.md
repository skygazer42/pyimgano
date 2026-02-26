# HST (Half-Space Trees)

Half-Space Trees (HST) is an ensemble-based anomaly detector that recursively partitions feature
space into half-spaces and uses the **mass distribution** across the partitions as an anomaly
signal.

In `pyimgano`:

- `core_hst`: operates on feature matrices
- `vision_hst`: vision wrapper using a `feature_extractor`

## When To Use

- You want a **fast** classical baseline with good behavior on moderately high-dimensional features
- You need a method that is less sensitive to kNN scaling than neighbor-based detectors

## Example (Tabular / Embeddings)

```python
import numpy as np
from pyimgano.models import create_model

X = np.random.default_rng(0).normal(size=(1000, 16))
det = create_model("core_hst", contamination=0.05, n_trees=25, max_depth=10, random_state=0)

det.fit(X)
scores = det.decision_function(X)
```

## Example (Vision Wrapper)

```python
from pyimgano.models import create_model

det = create_model(
    "vision_hst",
    contamination=0.05,
    n_trees=25,
    max_depth=10,
    feature_extractor={"name": "fft_lowfreq", "kwargs": {"resize_hw": [128, 128], "keep": 64}},
)
det.fit(train_images)
scores = det.decision_function(test_images)
```

## Notes

- For reproducibility, set `random_state`.
- HST is an ensemble; expect some variance on very small datasets.

