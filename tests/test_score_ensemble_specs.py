from __future__ import annotations

import numpy as np


def test_vision_score_ensemble_accepts_detector_specs() -> None:
    import pyimgano.models  # noqa: F401 - registry population
    from pyimgano.models import create_model

    rng = np.random.default_rng(0)
    x = [rng.normal(size=(8,)).astype(np.float32) for _ in range(80)]

    ens = create_model(
        "vision_score_ensemble",
        detectors=[
            {"name": "core_knn", "kwargs": {"n_neighbors": 5, "method": "largest"}},
            "core_ecod",
        ],
        contamination=0.1,
        combine="mean_rank",
    )
    ens.fit(x)
    scores = ens.decision_function(x[:10])
    preds = ens.predict(x[:10])

    assert scores.shape == (10,)
    assert preds.shape == (10,)
    assert np.all(np.isfinite(scores))
    assert set(np.unique(preds)).issubset({0, 1})
