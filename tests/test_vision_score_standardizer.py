from __future__ import annotations

import numpy as np


def test_vision_score_standardizer_rank_smoke_on_feature_vectors() -> None:
    """Vision wrapper should also work when inputs are already feature vectors.

    We use `feature_extractor='identity'` and pass vectors directly to keep this
    unit test fast and filesystem-free.
    """

    import pyimgano.models  # noqa: F401
    from pyimgano.models import create_model

    rng = np.random.default_rng(0)
    x = [rng.normal(size=(8,)).astype(np.float32) for _ in range(50)]

    det = create_model(
        "vision_score_standardizer",
        base_detector="vision_knn",
        base_kwargs={"feature_extractor": "identity", "n_neighbors": 5},
        method="rank",
        contamination=0.1,
    )
    det.fit(x)
    scores = det.decision_function(x[:10])
    preds = det.predict(x[:10])

    assert scores.shape == (10,)
    assert preds.shape == (10,)
    assert np.all(np.isfinite(scores))
    assert np.all(scores >= 0.0)
    assert np.all(scores <= 1.0)
    assert set(np.unique(preds)).issubset({0, 1})
