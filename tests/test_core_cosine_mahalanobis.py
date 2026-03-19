from __future__ import annotations

import numpy as np


def _unit_rows(x: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, float(eps))
    return (x / norms).astype(np.float64, copy=False)


def test_core_cosine_mahalanobis_smoke_and_direction() -> None:
    import pyimgano.models  # noqa: F401
    from pyimgano.models import create_model

    rng = np.random.default_rng(0)
    d = 6
    n_train = 80

    # Normal embeddings: near +e1 direction (unit vectors with small noise).
    base = np.zeros((n_train, d), dtype=np.float64)
    base[:, 0] = 1.0
    x_train = _unit_rows(base + 0.02 * rng.standard_normal(size=base.shape))

    # Inlier: +e1, Outlier: +e2 direction.
    x_inlier = _unit_rows(np.array([[1.0] + [0.0] * (d - 1)], dtype=np.float64))
    x_outlier = _unit_rows(np.array([[0.0, 1.0] + [0.0] * (d - 2)], dtype=np.float64))

    det = create_model(
        "core_cosine_mahalanobis",
        contamination=0.1,
        assume_centered=False,
        normalize=True,
    )
    det.fit(x_train)

    scores = np.asarray(
        det.decision_function(np.concatenate([x_inlier, x_outlier], axis=0)), dtype=np.float64
    )
    assert scores.shape == (2,)
    assert np.all(np.isfinite(scores))
    assert float(scores[1]) > float(scores[0])
