import numpy as np
import pytest


def test_detectors_module_is_importable() -> None:
    pytest.importorskip("pyod")

    from pyimgano import detectors

    assert hasattr(detectors, "IsolationForestDetector")


def test_isolation_forest_detector_compat_defaults_to_identity_extractor() -> None:
    pytest.importorskip("pyod")

    from pyimgano.detectors import IdentityFeatureExtractor, IsolationForestDetector

    detector = IsolationForestDetector(n_estimators=10, contamination=0.1)
    assert isinstance(detector.feature_extractor, IdentityFeatureExtractor)


def test_isolation_forest_detector_compat_fit_and_predict_proba_on_features() -> None:
    pytest.importorskip("pyod")

    from pyimgano.detectors import IsolationForestDetector

    X_train = np.random.randn(64, 8)
    X_test = np.random.randn(16, 8)

    detector = IsolationForestDetector(n_estimators=10, contamination=0.1)
    detector.fit(X_train)

    scores = np.asarray(detector.decision_function(X_test))
    assert scores.shape == (len(X_test),)

    proba = np.asarray(detector.predict_proba(X_test))
    assert proba.shape == (len(X_test), 2)
    assert np.all(proba >= 0.0)
    assert np.all(proba <= 1.0)

