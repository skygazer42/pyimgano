from __future__ import annotations

import numpy as np


def test_vision_suod_spec_accepts_model_specs() -> None:
    import pyimgano.models  # noqa: F401
    from pyimgano.models import create_model

    rng = np.random.default_rng(0)
    X = [rng.normal(size=(8,)).astype(np.float32) for _ in range(80)]

    det = create_model(
        "vision_suod_spec",
        feature_extractor="identity",
        contamination=0.1,
        base_estimator_specs=[
            {"name": "core_knn", "kwargs": {"n_neighbors": 5}},
            "core_ecod",
        ],
        combination="average",
        random_state=0,
    )
    det.fit(X)
    scores = det.decision_function(X[:10])
    preds = det.predict(X[:10])

    assert scores.shape == (10,)
    assert preds.shape == (10,)
    assert np.all(np.isfinite(scores))
    assert set(np.unique(preds)).issubset({0, 1})
