import numpy as np


def _write_png(path, *, value: int = 128) -> None:
    from PIL import Image

    img = np.ones((16, 16, 3), dtype=np.uint8) * int(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(str(path))


def test_sklearn_adapter_fit_score_predict(tmp_path) -> None:
    from pyimgano.sklearn_adapter import RegistryModelEstimator

    root = tmp_path / "ds"
    train = root / "train_0.png"
    test0 = root / "test_0.png"
    test1 = root / "test_1.png"
    _write_png(train, value=120)
    _write_png(test0, value=120)
    _write_png(test1, value=240)

    est = RegistryModelEstimator(model="vision_ecod", contamination=0.1)
    est.fit([str(train)])

    scores = est.decision_function([str(test0), str(test1)])
    assert isinstance(scores, np.ndarray)
    assert scores.shape == (2,)

    preds = est.predict([str(test0), str(test1)])
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (2,)
    assert set(np.unique(preds)).issubset({0, 1})


def test_sklearn_adapter_supports_clone() -> None:
    from sklearn.base import clone

    from pyimgano.sklearn_adapter import RegistryModelEstimator

    est = RegistryModelEstimator(model="vision_ecod", contamination=0.1)
    cloned = clone(est)
    assert isinstance(cloned, RegistryModelEstimator)
    assert cloned.get_params()["model"] == "vision_ecod"
    assert np.isclose(float(cloned.get_params()["contamination"]), 0.1)


def test_sklearn_adapter_exposes_confidence_and_rejection(monkeypatch) -> None:
    from pyimgano.sklearn_adapter import RegistryModelEstimator

    from pyimgano.models.base_detector import BaseDetector

    class DummyDetector(BaseDetector):
        def fit(self, X, y=None):  # noqa: ANN001
            self._set_n_classes(y)
            self.decision_scores_ = np.asarray(X, dtype=np.float64).reshape(-1)
            self._process_decision_scores()
            return self

        def decision_function(self, X):  # noqa: ANN001
            return np.asarray(X, dtype=np.float64).reshape(-1)

    monkeypatch.setattr(
        "pyimgano.models.registry.create_model",
        lambda model, **kwargs: DummyDetector(**kwargs),
    )
    monkeypatch.setattr("pyimgano.models.registry.list_models", lambda: ["dummy"])

    x_train = np.asarray([[0.0], [1.0], [2.0], [3.0], [4.0]], dtype=np.float32)
    x_test = np.asarray([[0.0], [2.0], [4.0]], dtype=np.float32)

    est = RegistryModelEstimator(model="dummy", contamination=0.2)
    est.fit(x_train)

    conf = est.predict_confidence(x_test)
    rejected = est.predict_with_rejection(x_test, confidence_threshold=0.75)

    conf_arr = np.asarray(conf, dtype=np.float64)
    rejected_arr = np.asarray(rejected, dtype=np.int64)
    assert conf_arr.shape == (3,)
    assert np.all(conf_arr >= 0.0)
    assert np.all(conf_arr <= 1.0)
    assert rejected_arr.tolist() == [0, -2, 1]
