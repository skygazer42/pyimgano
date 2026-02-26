from __future__ import annotations

import numpy as np


def test_core_loda_fit_predict_smoke() -> None:
    from pyimgano.models.loda import CoreLODA

    rng = np.random.RandomState(0)
    X = rng.normal(size=(60, 5)).astype(np.float32)

    det = CoreLODA(n_random_cuts=20, n_bins=10, contamination=0.2)
    det.fit(X)
    assert det.decision_scores_.shape == (60,)
    assert np.isfinite(det.decision_scores_).all()
    assert isinstance(det.threshold_, float)
    assert det.labels_.shape == (60,)
    assert set(np.unique(det.labels_)).issubset({0, 1})

    scores = det.decision_function(X[:9])
    assert scores.shape == (9,)
    assert np.isfinite(scores).all()

    pred = det.predict(X[:9])
    assert pred.shape == (9,)
    assert set(np.unique(pred)).issubset({0, 1})


def test_vision_loda_with_identity_extractor() -> None:
    from pyimgano.models import create_model

    class IdentityExtractor:
        def extract(self, X):
            return np.asarray(X)

    rng = np.random.RandomState(1)
    X = rng.normal(size=(30, 4)).astype(np.float32)

    det = create_model(
        "vision_loda",
        feature_extractor=IdentityExtractor(),
        n_random_cuts=10,
        n_bins=10,
        contamination=0.2,
    )
    det.fit(X)
    scores = det.decision_function(X[:5])
    assert scores.shape == (5,)
    assert np.isfinite(scores).all()

