import numpy as np


class _FakeInferencerDict:
    def __init__(self, scores_by_path: dict[str, float], maps_by_path: dict[str, np.ndarray]) -> None:
        self._scores_by_path = scores_by_path
        self._maps_by_path = maps_by_path

    def predict(self, image_path: str):
        return {
            "pred_score": float(self._scores_by_path[image_path]),
            "anomaly_map": self._maps_by_path[image_path],
        }


class _ObjResult:
    def __init__(self, *, pred_score: float, anomaly_map: np.ndarray) -> None:
        self.pred_score = float(pred_score)
        self.anomaly_map = anomaly_map


class _FakeInferencerObj:
    def __init__(self, scores_by_path: dict[str, float], maps_by_path: dict[str, np.ndarray]) -> None:
        self._scores_by_path = scores_by_path
        self._maps_by_path = maps_by_path

    def predict(self, image_path: str):
        return _ObjResult(
            pred_score=float(self._scores_by_path[image_path]),
            anomaly_map=self._maps_by_path[image_path],
        )


def test_anomalib_checkpoint_wrapper_calibrates_threshold_and_predicts():
    from pyimgano.models.anomalib_backend import VisionAnomalibCheckpoint

    train_paths = ["train_1.png", "train_2.png", "train_3.png", "train_4.png"]
    test_paths = ["test_low.png", "test_high.png"]

    scores = {
        "train_1.png": 0.10,
        "train_2.png": 0.20,
        "train_3.png": 0.30,
        "train_4.png": 0.40,
        "test_low.png": 0.15,
        "test_high.png": 0.90,
    }
    maps = {path: np.zeros((4, 4), dtype=np.float32) for path in scores}
    inferencer = _FakeInferencerDict(scores, maps)

    model = VisionAnomalibCheckpoint(
        checkpoint_path="ignored.ckpt",
        inferencer=inferencer,
        contamination=0.25,
        device="cpu",
    )
    model.fit(train_paths)

    expected_threshold = float(np.quantile([scores[p] for p in train_paths], 0.75))
    assert model.threshold_ == expected_threshold

    y_pred = model.predict(test_paths)
    assert set(np.unique(y_pred).tolist()).issubset({0, 1})
    assert y_pred.tolist() == [0, 1]


def test_anomalib_checkpoint_wrapper_returns_anomaly_maps():
    from pyimgano.models.anomalib_backend import VisionAnomalibCheckpoint

    paths = ["a.png", "b.png"]
    scores = {"a.png": 0.1, "b.png": 0.2}
    maps = {
        "a.png": np.ones((3, 5), dtype=np.float32),
        "b.png": np.zeros((3, 5), dtype=np.float32),
    }
    inferencer = _FakeInferencerObj(scores, maps)

    model = VisionAnomalibCheckpoint(
        checkpoint_path="ignored.ckpt",
        inferencer=inferencer,
        contamination=0.1,
        device="cpu",
    )

    m1 = model.get_anomaly_map("a.png")
    assert isinstance(m1, np.ndarray)
    assert m1.shape == (3, 5)
    assert float(m1.mean()) == 1.0

    stacked = model.predict_anomaly_map(paths)
    assert isinstance(stacked, np.ndarray)
    assert stacked.shape == (2, 3, 5)


def test_anomalib_checkpoint_wrapper_normalizes_map_shapes():
    from pyimgano.models.anomalib_backend import VisionAnomalibCheckpoint

    scores = {"a.png": 0.1, "b.png": 0.2}
    maps = {
        "a.png": np.ones((1, 3, 5), dtype=np.float32),
        "b.png": np.ones((3, 5, 1), dtype=np.float32),
    }
    inferencer = _FakeInferencerDict(scores, maps)

    model = VisionAnomalibCheckpoint(
        checkpoint_path="ignored.ckpt",
        inferencer=inferencer,
        contamination=0.1,
        device="cpu",
    )

    m1 = model.get_anomaly_map("a.png")
    assert m1.shape == (3, 5)
    assert m1.dtype == np.float32

    m2 = model.get_anomaly_map("b.png")
    assert m2.shape == (3, 5)
    assert m2.dtype == np.float32
