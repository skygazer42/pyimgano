# LoOP (Local Outlier Probability)

LoOP is a lightweight, neighbor-based detector that converts local distance deviation into a
**probability-like** outlier score in `[0, 1]` (higher means more anomalous).

In `pyimgano`, you can use it in two forms:

- `core_loop`: operates directly on a 2D feature matrix `(n_samples, n_features)`
- `vision_loop`: a vision wrapper that accepts image inputs and uses a `feature_extractor`

## When To Use

- You have a **small to medium** dataset (kNN-style methods scale poorly to very large `n`)
- You want a **bounded** score that is easy to interpret (`0..1`)
- Your anomalies are **local** (cluster-relative) rather than purely global

## Example (Tabular / Embeddings)

```python
import numpy as np
from pyimgano.models import create_model

X = np.random.default_rng(0).normal(size=(200, 32))

det = create_model("core_loop", contamination=0.05, n_neighbors=15, lambda_=3.0)
det.fit(X)
scores = det.decision_function(X)  # higher => more anomalous
labels = det.predict(X)            # 0/1 by fitted threshold
```

## Example (Vision Wrapper)

```python
from pyimgano.models import create_model

det = create_model(
    "vision_loop",
    contamination=0.05,
    n_neighbors=15,
    feature_extractor={"name": "hog", "kwargs": {"resize_hw": [128, 128]}},
)

# X can be paths or numpy images (uint8 HWC). See your extractor's contract.
det.fit(train_images)
scores = det.decision_function(test_images)
```

## Notes

- LoOP requires `n_samples > n_neighbors` at fit time.
- Scores are computed using a nearest-neighbors model; consider setting `n_jobs`
  where supported.

