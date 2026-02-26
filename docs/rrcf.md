# RRCF (Robust Random Cut Forest)

RRCF is a tree-based anomaly detector that builds a **random cut tree** over your feature vectors.
Points that are easy to separate by random cuts tend to have higher anomaly scores.

In `pyimgano`:

- `core_rrcf`: feature-matrix detector
- `vision_rrcf`: vision wrapper using a `feature_extractor`

## When To Use

- You want a **tree-based** classical baseline (often robust for mixed feature scales)
- You can afford a bit more compute than kNN, but still want a lightweight model

## Example (Tabular / Embeddings)

```python
import numpy as np
from pyimgano.models import create_model

X = np.random.default_rng(0).normal(size=(500, 32))
det = create_model("core_rrcf", contamination=0.05, n_trees=20, tree_size=256, random_state=0)
det.fit(X)
scores = det.decision_function(X)
```

## Example (Vision Wrapper)

```python
from pyimgano.models import create_model

det = create_model(
    "vision_rrcf",
    contamination=0.05,
    n_trees=20,
    tree_size=256,
    feature_extractor={"name": "patch_stats", "kwargs": {"grid_hw": [4, 4], "resize_hw": [128, 128]}},
)
det.fit(train_images)
scores = det.decision_function(test_images)
```

## Notes

- `tree_size` trades speed/memory for stability of the score.
- For reproducibility, set `random_state`.

