# ODIN (kNN In-Degree)

ODIN is a graph-based anomaly detector built on a simple idea:

> In a kNN graph, normal points are “popular” (high in-degree), while anomalies are rarely chosen
> as neighbors (low in-degree).

In `pyimgano`, you can use:

- `core_odin`: kNN in-degree on 2D features
- `vision_odin`: image/paths input via a `feature_extractor`

## When To Use

- You want a very fast-to-understand baseline
- Your anomalies are “孤立” points that are not similar to many others

## Example (Tabular / Embeddings)

```python
import numpy as np
from pyimgano.models import create_model

rng = np.random.default_rng(0)
X = rng.normal(size=(300, 8))
X[:5] += 8.0  # inject a few outliers

det = create_model("core_odin", contamination=0.05, n_neighbors=15)
det.fit(X)
scores = det.decision_function(X)
```

## Example (Vision Wrapper)

```python
from pyimgano.models import create_model

det = create_model(
    "vision_odin",
    contamination=0.05,
    n_neighbors=15,
    feature_extractor={"name": "hog", "kwargs": {"resize_hw": [96, 96]}},
)
det.fit(train_images)
scores = det.decision_function(test_images)
```

## Notes

- ODIN uses a kNN graph; `n_neighbors` is the key sensitivity knob.
- If many points are identical or highly duplicated, tie behavior can affect in-degree.

