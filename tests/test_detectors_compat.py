import numpy as np
import pytest


def test_detectors_module_is_importable() -> None:
    from pyimgano import detectors

    assert hasattr(detectors, "IsolationForestDetector")


def test_isolation_forest_detector_compat_defaults_to_identity_extractor() -> None:
    from pyimgano.detectors import IdentityFeatureExtractor, IsolationForestDetector

    detector = IsolationForestDetector(n_estimators=10, contamination=0.1)
    assert isinstance(detector.feature_extractor, IdentityFeatureExtractor)


def test_isolation_forest_detector_compat_fit_and_predict_proba_on_features() -> None:
    from pyimgano.detectors import IsolationForestDetector

    rng = np.random.default_rng(0)
    X_train = rng.standard_normal((64, 8))
    X_test = rng.standard_normal((16, 8))

    detector = IsolationForestDetector(n_estimators=10, contamination=0.1)
    detector.fit(X_train)

    scores = np.asarray(detector.decision_function(X_test))
    assert scores.shape == (len(X_test),)

    proba = np.asarray(detector.predict_proba(X_test))
    assert proba.shape == (len(X_test), 2)
    assert np.all(proba >= 0.0)
    assert np.all(proba <= 1.0)


def test_registry_model_estimator_smoke_on_feature_vectors() -> None:
    from pyimgano.sklearn_adapter import RegistryModelEstimator

    class IdentityExtractor:
        def extract(self, X):
            return np.asarray(X)

    rng = np.random.default_rng(0)
    X_train = rng.standard_normal((32, 8)).astype(np.float32)
    X_test = rng.standard_normal((8, 8)).astype(np.float32)

    est = RegistryModelEstimator(
        model="vision_ecod",
        feature_extractor=IdentityExtractor(),
        contamination=0.1,
    )
    est.fit(X_train)

    scores = np.asarray(est.decision_function(X_test))
    assert scores.shape == (len(X_test),)
    assert np.isfinite(scores).all()

    preds = np.asarray(est.predict(X_test), dtype=int)
    assert preds.shape == (len(X_test),)
    assert set(np.unique(preds)).issubset({0, 1})


def test_registry_model_estimator_errors_are_clear() -> None:
    from sklearn.exceptions import NotFittedError

    from pyimgano.sklearn_adapter import RegistryModelEstimator

    class IdentityExtractor:
        def extract(self, X):
            return np.asarray(X)

    est = RegistryModelEstimator(
        model="vision_ecod",
        feature_extractor=IdentityExtractor(),
        contamination=0.1,
    )

    with pytest.raises(NotFittedError):
        est.decision_function(np.random.default_rng(0).standard_normal((2, 3)))

    with pytest.raises(TypeError, match="single path-like"):
        est.fit("train_0.png")

    with pytest.raises(ValueError, match="Unknown model name"):
        RegistryModelEstimator(model="__does_not_exist__").fit(np.random.randn(4, 2))
