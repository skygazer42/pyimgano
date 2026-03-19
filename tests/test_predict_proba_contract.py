import numpy as np
import pytest


def test_predict_proba_works_for_vision_iforest_on_features() -> None:
    from pyimgano.models import create_model

    class IdentityExtractor:
        def extract(self, x):
            return np.asarray(x)

    rng = np.random.default_rng(0)
    x_train = rng.standard_normal((64, 8))
    x_test = rng.standard_normal((16, 8))

    detector = create_model(
        "vision_iforest",
        feature_extractor=IdentityExtractor(),
        contamination=0.1,
        n_estimators=10,
    )
    detector.fit(x_train)

    proba = np.asarray(detector.predict_proba(x_test))
    assert proba.shape == (len(x_test), 2)
    assert np.all(proba >= 0.0)
    assert np.all(proba <= 1.0)
