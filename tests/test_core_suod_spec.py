from __future__ import annotations

import numpy as np


def test_core_suod_spec_fit_predict_smoke() -> None:
    import pyimgano.models  # noqa: F401 - registry population
    from pyimgano.models import create_model

    rng = np.random.default_rng(0)
    x = rng.normal(size=(120, 8)).astype(np.float64)

    det = create_model(
        "core_suod_spec",
        base_estimator_specs=[
            {"name": "core_knn", "kwargs": {"n_neighbors": 10, "method": "largest"}},
            "core_ecod",
            {"name": "core_iforest", "kwargs": {"n_estimators": 50, "random_state": 0}},
        ],
        contamination=0.1,
        combination="average",
        standardize="zscore",
        random_state=0,
    )
    det.fit(x)

    scores = det.decision_function(x[:10])
    preds = det.predict(x[:10])

    assert hasattr(det, "decision_scores_")
    assert np.asarray(det.decision_scores_).shape == (120,)
    assert scores.shape == (10,)
    assert preds.shape == (10,)
    assert np.all(np.isfinite(scores))
    assert set(np.unique(preds)).issubset({0, 1})
