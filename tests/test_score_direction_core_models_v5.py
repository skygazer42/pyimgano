from __future__ import annotations

import numpy as np


def _unit_rows(x: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, float(eps))
    return (x / norms).astype(np.float64, copy=False)


def test_score_direction_core_kde_ratio_outlier_higher() -> None:
    import pyimgano.models  # noqa: F401
    from pyimgano.models import create_model

    rng = np.random.default_rng(0)
    X_train = rng.normal(loc=0.0, scale=1.0, size=(256, 4)).astype(np.float64)
    x_inlier = np.zeros((1, 4), dtype=np.float64)
    x_outlier = np.full((1, 4), 8.0, dtype=np.float64)

    det = create_model("core_kde_ratio", contamination=0.1, bandwidth_local=0.2, bandwidth_global=1.0)
    det.fit(X_train)
    s = np.asarray(det.decision_function(np.concatenate([x_inlier, x_outlier], axis=0)), dtype=np.float64)
    assert s.shape == (2,)
    assert np.all(np.isfinite(s))
    assert float(s[1]) > float(s[0])


def test_score_direction_core_knn_cosine_outlier_higher() -> None:
    import pyimgano.models  # noqa: F401
    from pyimgano.models import create_model

    rng = np.random.default_rng(0)
    d = 8
    n_train = 96

    base = np.zeros((n_train, d), dtype=np.float64)
    base[:, 0] = 1.0
    X_train = _unit_rows(base + 0.04 * rng.standard_normal(size=base.shape))

    x_inlier = _unit_rows(np.array([[1.0] + [0.0] * (d - 1)], dtype=np.float64))
    x_outlier = _unit_rows(np.array([[-1.0] + [0.0] * (d - 1)], dtype=np.float64))

    det = create_model("core_knn_cosine", contamination=0.1, n_neighbors=5, method="largest", normalize=True)
    det.fit(X_train)
    s = np.asarray(det.decision_function(np.concatenate([x_inlier, x_outlier], axis=0)), dtype=np.float64)
    assert s.shape == (2,)
    assert np.all(np.isfinite(s))
    assert float(s[1]) > float(s[0])

