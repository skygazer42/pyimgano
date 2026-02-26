from __future__ import annotations

import numpy as np


def test_core_deep_svdd_smoke_can_fit_and_score() -> None:
    from pyimgano.models.deep_svdd import CoreDeepSVDD

    rng = np.random.RandomState(0)
    X = rng.normal(size=(20, 8)).astype(np.float32)

    det = CoreDeepSVDD(
        n_features=8,
        hidden_neurons=[16, 8],
        use_autoencoder=True,
        epochs=1,
        batch_size=4,
        verbose=0,
        random_state=0,
        contamination=0.2,
    )
    det.fit(X)

    scores = det.decision_function(X[:5])
    assert scores.shape == (5,)
    assert np.isfinite(scores).all()

    labels = det.predict(X[:5])
    assert set(np.unique(labels)).issubset({0, 1})


def test_vision_deep_svdd_smoke_can_fit_with_dummy_features() -> None:
    from pyimgano.models import create_model

    class DummyFeatureExtractor:
        def __init__(self, feature_dim: int = 8) -> None:
            self.feature_dim = int(feature_dim)

        def extract(self, inputs):
            inputs = list(inputs)
            rng = np.random.RandomState(123)
            return rng.normal(size=(len(inputs), self.feature_dim)).astype(np.float32)

    det = create_model(
        "vision_deep_svdd",
        feature_extractor=DummyFeatureExtractor(feature_dim=8),
        n_features=8,
        hidden_neurons=[16, 8],
        use_autoencoder=True,
        epochs=1,
        batch_size=2,
        verbose=0,
        random_state=0,
        contamination=0.2,
    )

    train = ["a.png", "b.png", "c.png", "d.png"]
    test = ["e.png", "f.png", "g.png"]
    det.fit(train)
    scores = det.decision_function(test)
    assert scores.shape == (len(test),)
    assert np.isfinite(scores).all()

