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
    x_train = rng.standard_normal((64, 8))
    x_test = rng.standard_normal((16, 8))

    detector = IsolationForestDetector(n_estimators=10, contamination=0.1)
    detector.fit(x_train)

    scores = np.asarray(detector.decision_function(x_test))
    assert scores.shape == (len(x_test),)

    proba = np.asarray(detector.predict_proba(x_test))
    assert proba.shape == (len(x_test), 2)
    assert np.all(proba >= 0.0)
    assert np.all(proba <= 1.0)


def test_registry_model_estimator_smoke_on_feature_vectors() -> None:
    from pyimgano.sklearn_adapter import RegistryModelEstimator

    class IdentityExtractor:
        def extract(self, x):
            return np.asarray(x)

    rng = np.random.default_rng(0)
    x_train = rng.standard_normal((32, 8)).astype(np.float32)
    x_test = rng.standard_normal((8, 8)).astype(np.float32)

    est = RegistryModelEstimator(
        model="vision_ecod",
        feature_extractor=IdentityExtractor(),
        contamination=0.1,
    )
    est.fit(x_train)

    scores = np.asarray(est.decision_function(x_test))
    assert scores.shape == (len(x_test),)
    assert np.isfinite(scores).all()

    preds = np.asarray(est.predict(x_test), dtype=int)
    assert preds.shape == (len(x_test),)
    assert set(np.unique(preds)).issubset({0, 1})


def test_registry_model_estimator_errors_are_clear() -> None:
    from sklearn.exceptions import NotFittedError

    from pyimgano.sklearn_adapter import RegistryModelEstimator

    class IdentityExtractor:
        def extract(self, x):
            return np.asarray(x)

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
        RegistryModelEstimator(model="__does_not_exist__").fit(
            np.random.default_rng(0).standard_normal((4, 2))
        )
