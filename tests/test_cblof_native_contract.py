from __future__ import annotations

import numpy as np


def test_core_cblof_fit_predict_smoke() -> None:
    from pyimgano.models.cblof import CoreCBLOF

    rng = np.random.default_rng(0)
    x = rng.normal(size=(50, 4)).astype(np.float32)

    det = CoreCBLOF(n_clusters=3, contamination=0.2, random_state=0)
    det.fit(x)

    assert det.decision_scores_.shape == (50,)
    assert np.isfinite(det.decision_scores_).all()
    assert isinstance(det.threshold_, float)
    assert det.labels_.shape == (50,)
    assert set(np.unique(det.labels_)).issubset({0, 1})

    scores = det.decision_function(x[:7])
    assert scores.shape == (7,)
    assert np.isfinite(scores).all()

    pred = det.predict(x[:7])
    assert pred.shape == (7,)
    assert set(np.unique(pred)).issubset({0, 1})


def test_vision_cblof_with_identity_extractor() -> None:
    from pyimgano.models import create_model

    class IdentityExtractor:
        def extract(self, x):
            return np.asarray(x)

    rng = np.random.default_rng(1)
    x = rng.normal(size=(40, 3)).astype(np.float32)

    det = create_model(
        "vision_cblof",
        feature_extractor=IdentityExtractor(),
        n_clusters=3,
        contamination=0.25,
        random_state=0,
    )
    det.fit(x)
    scores = det.decision_function(x[:5])
    assert scores.shape == (5,)
    assert np.isfinite(scores).all()
