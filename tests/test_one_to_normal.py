from __future__ import annotations

import numpy as np

import pyimgano.models as models


class _FakeNormalizer:
    def normalize(self, image: np.ndarray) -> np.ndarray:
        arr = np.asarray(image, dtype=np.float32)
        return arr * 0.1


def test_one_to_normal_scores_residual_and_emits_maps() -> None:
    detector = models.create_model(
        "vision_one_to_normal",
        normalizer=_FakeNormalizer(),
        contamination=0.25,
    )

    normal = np.zeros((4, 4), dtype=np.float32)
    anomaly = np.zeros((4, 4), dtype=np.float32)
    anomaly[1:3, 1:3] = 5.0

    detector.fit([normal, anomaly])
    scores = np.asarray(detector.decision_function([normal, anomaly]), dtype=np.float64)
    assert scores.shape == (2,)
    assert float(scores[1]) > float(scores[0])

    maps = np.asarray(detector.predict_anomaly_map([normal, anomaly]), dtype=np.float32)
    assert maps.shape == (2, 4, 4)
    assert float(np.max(maps[1])) > float(np.max(maps[0]))
