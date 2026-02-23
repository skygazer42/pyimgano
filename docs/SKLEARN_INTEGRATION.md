# scikit-learn Integration

PyImgAno aims to be usable in both:

- **Industrial pipelines** (CLI runners + run artifacts)
- **Estimator-driven workflows** (sklearn / PyOD conventions)

This document shows how to use PyImgAno detectors in scikit-learn-style code.

---

## Registry → sklearn Estimator

Use `pyimgano.sklearn_adapter.RegistryModelEstimator` to wrap a registry model
as a scikit-learn compatible estimator.

```python
from pyimgano.sklearn_adapter import RegistryModelEstimator

est = RegistryModelEstimator(
    model="vision_ecod",
    contamination=0.1,
)
```

### Fit / Score / Predict

```python
train_paths = ["train_0.png", "train_1.png"]
test_paths = ["test_0.png", "test_1.png"]

est.fit(train_paths)

# Continuous anomaly scores (higher = more anomalous)
scores = est.decision_function(test_paths)

# Binary labels: 0 = normal, 1 = anomaly
preds = est.predict(test_paths)
```

Notes:

- The adapter constructs the underlying detector inside `fit()`.
- `decision_function()` always returns a 1D numpy array of shape `(N,)`.
- `predict()` normalizes predictions to `{0, 1}` (PyOD-style).

---

## `sklearn.base.clone` Support

The adapter provides `get_params/set_params` so that `sklearn.base.clone()` can
reconstruct the estimator and reuse it in search utilities.

```python
from sklearn.base import clone

est2 = clone(est)
assert est2.get_params()["model"] == "vision_ecod"
```

---

## Working With sklearn Metrics

Most sklearn metrics expect:

- `y_true` in `{0, 1}`
- `y_score` as continuous scores

Example (AUROC):

```python
from sklearn.metrics import roc_auc_score

auroc = roc_auc_score(y_true, scores)
```

---

## Limitations / Tips

- Not every deep model is a good fit for sklearn-style cross-validation (GPU,
  large checkpoint downloads, long training time).
- For production benchmarking + run artifacts, prefer `pyimgano-benchmark`.
- For strict sklearn `OutlierMixin` conventions (inliers=1, outliers=-1),
  convert `preds` yourself if needed.

