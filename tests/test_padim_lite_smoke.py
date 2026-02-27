from __future__ import annotations

import numpy as np


def test_vision_padim_lite_smoke_on_vectors() -> None:
    import pyimgano.models  # noqa: F401
    from pyimgano.models import create_model

    rng = np.random.RandomState(0)
    X = [rng.normal(size=(10,)).astype(np.float32) for _ in range(80)]

    det = create_model(
        "vision_padim_lite",
        feature_extractor="identity",
        contamination=0.1,
        robust=False,
    )
    det.fit(X)
    scores = det.decision_function(X[:10])
    preds = det.predict(X[:10])

    assert scores.shape == (10,)
    assert preds.shape == (10,)
    assert np.all(np.isfinite(scores))
    assert set(np.unique(preds)).issubset({0, 1})

