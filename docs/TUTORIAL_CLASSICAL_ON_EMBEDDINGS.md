# Tutorial: Classical Anomaly Detection On Embeddings

This tutorial shows how to run **classical** anomaly detectors on **precomputed feature vectors**
(a.k.a. embeddings) such as:

- CNN / ViT global pooled features
- Patch embedding aggregates
- Any custom descriptors you already compute upstream

The key idea is:

> Classical detectors operate on a 2D matrix `X` with shape `(n_samples, n_features)`.

## 1. Use `core_*` Detectors Directly (Recommended For Embeddings)

```python
import numpy as np
from pyimgano.models import create_model

# Example embedding matrix (replace with your real embeddings)
X_train = np.random.default_rng(0).normal(size=(500, 128))
X_test = np.random.default_rng(1).normal(size=(100, 128))

det = create_model("core_ecod", contamination=0.05)
det.fit(X_train)
scores = det.decision_function(X_test)
labels = det.predict(X_test)
```

Good starting points:

- `core_ecod`, `core_copod` (parameter-light baselines)
- `core_mahalanobis`, `core_pca_md`, `core_rzscore` (strong sanity-check baselines)
- `core_loop`, `core_ldof`, `core_odin` (local neighborhood geometry)
- `core_rrcf`, `core_hst` (tree/ensemble baselines)

## 2. Use `vision_*` Wrappers With `identity` Extractor (Sklearn-Style Ergonomics)

If you already have a workflow that expects `vision_*` models, you can still run them on embeddings
by using the built-in `identity` feature extractor.

```python
import numpy as np
from pyimgano.models import create_model

X_train = np.random.default_rng(0).normal(size=(500, 128))
X_test = np.random.default_rng(1).normal(size=(100, 128))

det = create_model(
    "vision_ecod",
    contamination=0.05,
    feature_extractor="identity",  # or {"name": "identity"}
)
det.fit(X_train)
scores = det.decision_function(X_test)
```

## 3. Add Simple Feature Processing (Scaling / PCA)

You can combine multiple extractors using `multi`:

```python
from pyimgano.models import create_model

det = create_model(
    "vision_mahalanobis",
    contamination=0.05,
    feature_extractor={
        "name": "multi",
        "kwargs": {
            "parts": [
                {"name": "scaler", "kwargs": {}},
                {"name": "pca_projector", "kwargs": {"n_components": 64}},
            ]
        },
    },
)
det.fit(X_train)
scores = det.decision_function(X_test)
```

## 4. Advanced: Dynamic Pipelines With `vision_feature_pipeline`

Use this when you want to pair *any* registered `core_*` detector with *any* registered feature
extractor via a JSON-friendly config:

```python
from pyimgano.models import create_model

det = create_model(
    "vision_feature_pipeline",
    contamination=0.05,
    core_detector="core_loop",
    core_kwargs={"n_neighbors": 15},
    feature_extractor="identity",
)
det.fit(X_train)
scores = det.decision_function(X_test)
```

## Notes

- All detectors follow the same scoring convention: **higher score = more anomalous**.
- For large datasets, prefer `core_ecod`/`core_copod`-style baselines, or pre-reduce your
  embeddings (e.g. PCA).

