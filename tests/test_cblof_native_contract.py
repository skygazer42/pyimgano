from __future__ import annotations

import numpy as np


def test_core_cblof_fit_predict_smoke() -> None:
    from pyimgano.models.cblof import CoreCBLOF

    rng = np.random.RandomState(0)
    X = rng.normal(size=(50, 4)).astype(np.float32)

    det = CoreCBLOF(n_clusters=3, contamination=0.2, random_state=0)
    det.fit(X)

    assert det.decision_scores_.shape == (50,)
    assert np.isfinite(det.decision_scores_).all()
    assert isinstance(det.threshold_, float)
    assert det.labels_.shape == (50,)
    assert set(np.unique(det.labels_)).issubset({0, 1})

    scores = det.decision_function(X[:7])
    assert scores.shape == (7,)
    assert np.isfinite(scores).all()

    pred = det.predict(X[:7])
    assert pred.shape == (7,)
    assert set(np.unique(pred)).issubset({0, 1})


def test_vision_cblof_with_identity_extractor() -> None:
    from pyimgano.models import create_model

    class IdentityExtractor:
        def extract(self, X):
            return np.asarray(X)

    rng = np.random.RandomState(1)
    X = rng.normal(size=(40, 3)).astype(np.float32)

    det = create_model(
        "vision_cblof",
        feature_extractor=IdentityExtractor(),
        n_clusters=3,
        contamination=0.25,
        random_state=0,
    )
    det.fit(X)
    scores = det.decision_function(X[:5])
    assert scores.shape == (5,)
    assert np.isfinite(scores).all()

