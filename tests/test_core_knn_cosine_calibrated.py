from __future__ import annotations

import numpy as np


def _unit_rows(x: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, float(eps))
    return (x / norms).astype(np.float64, copy=False)


def test_core_knn_cosine_calibrated_rank_smoke_and_direction() -> None:
    import pyimgano.models  # noqa: F401
    from pyimgano.models import create_model

    rng = np.random.default_rng(0)
    d = 8
    n_train = 64

    # Train normals are near +e1 direction.
    base = np.zeros((n_train, d), dtype=np.float64)
    base[:, 0] = 1.0
    X_train = base + 0.03 * rng.standard_normal(size=base.shape)
    X_train = _unit_rows(X_train)

    x_inlier = _unit_rows(np.array([[1.0] + [0.0] * (d - 1)], dtype=np.float64))
    x_outlier = _unit_rows(np.array([[-1.0] + [0.0] * (d - 1)], dtype=np.float64))

    det = create_model(
        "core_knn_cosine_calibrated",
        contamination=0.1,
        n_neighbors=5,
        knn_method="largest",
        method="rank",
    )
    det.fit(X_train)

    scores = np.asarray(det.decision_function(np.concatenate([x_inlier, x_outlier], axis=0)), dtype=np.float64)
    assert scores.shape == (2,)
    assert np.all(np.isfinite(scores))
    assert float(scores[1]) > float(scores[0])
    assert np.all((scores >= 0.0) & (scores <= 1.0))

