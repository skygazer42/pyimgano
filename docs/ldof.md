# LDOF (Local Distance-based Outlier Factor)

LDOF measures how far a point is from its neighbors **relative** to the neighbor-neighbor distances.
This makes it a useful baseline for clustered data where “global distance” methods can be misleading.

In `pyimgano`, you can use:

- `core_ldof`: 2D feature matrix input
- `vision_ldof`: image/paths input via a `feature_extractor`

## When To Use

- Your data has **clusters** / multiple modes
- You want a simple, explainable local baseline (kNN graph geometry)

## Example (Tabular / Embeddings)

```python
import numpy as np
from pyimgano.models import create_model

X = np.random.default_rng(0).normal(size=(300, 16))
det = create_model("core_ldof", contamination=0.05, n_neighbors=20)

det.fit(X)
scores = det.decision_function(X)
```

## Example (Vision Wrapper)

```python
from pyimgano.models import create_model

det = create_model(
    "vision_ldof",
    contamination=0.05,
    n_neighbors=20,
    feature_extractor={"name": "lbp", "kwargs": {"n_points": 8, "radius": 1.0}},
)

det.fit(train_images)
scores = det.decision_function(test_images)
```

## Notes

- Complexity is kNN-driven; treat it as a **classical baseline**, not a huge-scale solution.

