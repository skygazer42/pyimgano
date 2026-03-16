from __future__ import annotations

import numpy as np


def test_vision_lscp_spec_accepts_model_specs() -> None:
    import pyimgano.models  # noqa: F401
    from pyimgano.models import create_model

    rng = np.random.default_rng(0)
    X = [rng.normal(size=(8,)).astype(np.float32) for _ in range(80)]

    det = create_model(
        "vision_lscp_spec",
        feature_extractor="identity",
        contamination=0.1,
        detector_specs=[
            "core_ecod",
            {"name": "core_knn", "kwargs": {"n_neighbors": 7, "method": "largest"}},
        ],
        local_region_size=15,
        n_bins=5,
        random_state=0,
    )
    det.fit(X)
    scores = det.decision_function(X[:10])
    preds = det.predict(X[:10])

    assert scores.shape == (10,)
    assert preds.shape == (10,)
    assert np.all(np.isfinite(scores))
    assert set(np.unique(preds)).issubset({0, 1})
