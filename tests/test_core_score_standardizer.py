from __future__ import annotations

import numpy as np


def test_core_score_standardizer_rank_smoke() -> None:
    import pyimgano.models  # noqa: F401
    from pyimgano.models import create_model

    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 6))

    det = create_model(
        "core_score_standardizer",
        base_detector="core_knn",
        base_kwargs={"n_neighbors": 5, "method": "largest"},
        method="rank",
        contamination=0.1,
    )
    det.fit(X)
    scores = det.decision_function(X[:10])
    preds = det.predict(X[:10])

    assert scores.shape == (10,)
    assert preds.shape == (10,)
    assert np.all(np.isfinite(scores))
    assert np.all(scores >= 0.0)
    assert np.all(scores <= 1.0)
    assert set(np.unique(preds)).issubset({0, 1})
