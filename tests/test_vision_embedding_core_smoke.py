from __future__ import annotations

import numpy as np


def test_vision_embedding_core_smoke_on_vectors() -> None:
    import pyimgano.models  # noqa: F401
    from pyimgano.models import create_model

    rng = np.random.default_rng(0)
    x = [rng.normal(size=(12,)).astype(np.float32) for _ in range(80)]

    det = create_model(
        "vision_embedding_core",
        contamination=0.1,
        embedding_extractor="identity",
        core_detector="core_ecod",
    )
    det.fit(x)
    scores = det.decision_function(x[:10])
    preds = det.predict(x[:10])

    assert scores.shape == (10,)
    assert preds.shape == (10,)
    assert np.all(np.isfinite(scores))
    assert set(np.unique(preds)).issubset({0, 1})
