import numpy as np
import pytest


def test_predict_proba_works_for_vision_iforest_on_features() -> None:
    pytest.importorskip("pyod")

    from pyimgano.models import create_model

    class IdentityExtractor:
        def extract(self, X):
            return np.asarray(X)

    X_train = np.random.randn(64, 8)
    X_test = np.random.randn(16, 8)

    detector = create_model(
        "vision_iforest",
        feature_extractor=IdentityExtractor(),
        contamination=0.1,
        n_estimators=10,
    )
    detector.fit(X_train)

    proba = np.asarray(detector.predict_proba(X_test))
    assert proba.shape == (len(X_test), 2)
    assert np.all(proba >= 0.0)
    assert np.all(proba <= 1.0)

