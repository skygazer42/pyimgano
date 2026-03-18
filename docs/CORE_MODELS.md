# Classical `core_*` Models

`core_*` models are the lowest-friction way to run classical anomaly detection
on a 2D feature matrix:

- input: `X.shape == (n_samples, n_features)`
- output: anomaly scores where **higher means more anomalous**

Use them when you already have embeddings, tabular descriptors, or handcrafted
image features and do not need a path-oriented vision wrapper.

## When To Reach For `core_*`

Good fits:

- precomputed embeddings from a CNN, ViT, ONNX model, or TorchScript model
- handcrafted image descriptors such as HOG, LBP, FFT, patch stats
- tabular or sensor features that already live in memory
- industrial pipelines where you want a small, auditable scoring layer on top
  of a separate feature extractor

Use `vision_*` or pixel-map models instead when:

- you want the detector to read image paths directly
- you need anomaly maps / defect masks
- you want end-to-end image preprocessing baked into the model wrapper

## Good Starting Points

| Goal | Start with | Why |
|---|---|---|
| Strong default baseline | `core_ecod` | parameter-light, robust, fast |
| Proven general-purpose baseline | `core_iforest` | widely understood and stable |
| Local neighborhood geometry | `core_knn`, `core_lof`, `core_loop` | useful for manifold-style embeddings |
| Robust statistical sanity check | `core_mahalanobis`, `core_mcd`, `core_rzscore` | simple audit-friendly baselines |
| Density contrast | `core_kde`, `core_kde_ratio` | useful when density mismatch matters |
| Clustering-style distance | `core_kmeans`, `core_dbscan` | cheap baselines for compact feature spaces |
| Ensemble / score calibration | `core_lscp_spec`, `core_suod_spec`, `core_score_standardizer` | JSON-friendly composition |

## Direct Python Usage

```python
import numpy as np

from pyimgano.models import create_model

rng = np.random.default_rng(0)
X_train = rng.normal(size=(64, 128)).astype(np.float32)
X_test = rng.normal(size=(8, 128)).astype(np.float32)

det = create_model("core_knn", contamination=0.1, n_neighbors=5)
det.fit(X_train)
scores = det.decision_function(X_test)
labels = det.predict(X_test)
```

Another common baseline:

```python
from pyimgano.models import create_model

det = create_model("core_iforest", contamination=0.05, n_estimators=200, random_state=0)
det.fit(X_train)
scores = det.decision_function(X_test)
labels, confidence = det.predict(X_test, return_confidence=True)
labels_with_reject = det.predict_with_rejection(X_test, confidence_threshold=0.75)
```

## Confidence And Rejection

Native `core_*` detectors that inherit the shared `BaseDetector` contract now expose:

- `predict(X, return_confidence=True)` → `(labels, confidence)`
- `predict_confidence(X)` → confidence of the predicted label on `[0, 1]`
- `predict_with_rejection(X, confidence_threshold=..., reject_label=-2)` → low-confidence samples are marked as rejected

Practical semantics:

- confidence is high when a sample is far into the predicted side of the training score distribution
- confidence is lower near the decision threshold
- rejected samples use `-2` by default so they stay distinct from inlier `0` and outlier `1`

## Composition Patterns

### 1. `vision_feature_pipeline`

Use this when you want a registered feature extractor plus a `core_*` detector
without writing a dedicated model class.

```python
from pyimgano.models import create_model

pipe = create_model(
    "vision_feature_pipeline",
    feature_extractor={"name": "hog", "kwargs": {"resize_hw": [128, 128]}},
    core_detector="core_ecod",
    core_kwargs={},
    contamination=0.1,
)
```

### 2. `vision_embedding_core`

Use this when your feature stage is a deep embedding extractor such as
`torchvision_backbone`, `torchscript_embed`, or `onnx_embed`.

```python
from pyimgano.models import create_model

det = create_model(
    "vision_embedding_core",
    embedding_extractor="torchvision_backbone",
    embedding_kwargs={"backbone": "resnet18", "pretrained": False, "device": "cpu"},
    core_detector="core_ecod",
    core_kwargs={},
    contamination=0.1,
)
```

## Discovery

From the CLI:

```bash
pyimgano-benchmark --list-models --tags core
pyimgano-benchmark --model-info core_knn
pyimgano-benchmark --model-info core_iforest --json
```

From Python:

```python
from pyimgano.models import list_models

print(list_models(tags=["core"])[:20])
```

## Operational Notes

- `core_*` detectors do not generate pixel maps.
- `threshold_` is still part of the detector contract, so `predict()` works the
  same way as other models.
- For industrial deployment, keep the feature extractor and the `core_*`
  detector separate in your mental model: the extractor defines representation
  quality; the core model defines anomaly scoring.

See also:

- `docs/TUTORIAL_CLASSICAL_ON_EMBEDDINGS.md`
- `docs/TUTORIAL_EMBEDDINGS_PLUS_CORE.md`
- `docs/FEATURE_EXTRACTORS.md`
