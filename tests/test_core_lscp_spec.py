from __future__ import annotations

import numpy as np


def test_core_lscp_spec_fit_predict_smoke() -> None:
    import pyimgano.models  # noqa: F401 - registry population
    from pyimgano.models import create_model

    rng = np.random.RandomState(0)
    X = rng.normal(size=(120, 8)).astype(np.float64)

    det = create_model(
        "core_lscp_spec",
        detector_specs=[
            "core_ecod",
            {"name": "core_hbos", "kwargs": {"n_bins": 10}},
        ],
        contamination=0.1,
        local_region_size=20,
        local_region_iterations=5,
        local_min_features=0.5,
        local_max_features=1.0,
        n_bins=10,
        random_state=0,
    )
    det.fit(X)

    scores = det.decision_function(X[:10])
    preds = det.predict(X[:10])

    assert hasattr(det, "decision_scores_")
    assert np.asarray(det.decision_scores_).shape == (120,)
    assert scores.shape == (10,)
    assert preds.shape == (10,)
    assert np.all(np.isfinite(scores))
    assert set(np.unique(preds)).issubset({0, 1})

