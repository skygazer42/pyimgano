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
    assert float(cloned.get_params()["contamination"]) == 0.1
