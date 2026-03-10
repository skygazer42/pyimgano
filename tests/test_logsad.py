from __future__ import annotations

import numpy as np

import pyimgano.models as models


class _FakeVisualBackend:
    def score(self, image: np.ndarray):
        arr = np.asarray(image, dtype=np.float32)
        anomaly_map = arr.astype(np.float32)
        return float(np.max(anomaly_map)), anomaly_map


class _FakeLogicBackend:
    def score(self, image: np.ndarray) -> float:
        arr = np.asarray(image, dtype=np.float32)
        return float(np.mean(arr > 0.0))


def test_logsad_combines_visual_and_logic_scores_and_emits_maps() -> None:
    detector = models.create_model(
        "vision_logsad",
        visual_backend=_FakeVisualBackend(),
        logic_backend=_FakeLogicBackend(),
        contamination=0.25,
    )

    support = [np.zeros((2, 2), dtype=np.float32)]
    normal = np.zeros((2, 2), dtype=np.float32)
    anomaly = np.array([[0.0, 0.0], [0.0, 5.0]], dtype=np.float32)

    detector.fit(support)

    scores = np.asarray(detector.decision_function([normal, anomaly]), dtype=np.float64)
    maps = np.asarray(detector.predict_anomaly_map([normal, anomaly]), dtype=np.float32)
    labels = np.asarray(detector.predict([normal, anomaly]), dtype=np.int64)

    assert scores.shape == (2,)
    assert float(scores[1]) > float(scores[0])
    assert maps.shape == (2, 2, 2)
    assert float(np.max(maps[1])) > float(np.max(maps[0]))
    assert labels.tolist() == [0, 1]
