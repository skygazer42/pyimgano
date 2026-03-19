from __future__ import annotations

import numpy as np

from pyimgano.inference.runtime_adapter import score_and_maps


class _TupleDetector:
    input_mode = "numpy"

    def decision_function(self, x):
        n = len(list(x))
        scores = np.arange(float(n), dtype=np.float32)
        maps = [np.full((3, 3), fill_value=i, dtype=np.float32) for i in range(n)]
        return scores, maps


class _SingleMapDetector:
    input_mode = "numpy"

    def decision_function(self, x):
        n = len(list(x))
        return np.linspace(0.0, 1.0, num=n, dtype=np.float32)

    def get_anomaly_map(self, x):
        del x
        return np.ones((3, 3), dtype=np.float32)


class _PredictMap2DDetector:
    input_mode = "numpy"

    def decision_function(self, x):
        n = len(list(x))
        return np.linspace(0.0, 1.0, num=n, dtype=np.float32)

    def predict_anomaly_map(self, x):
        _ = x
        return np.full((4, 4), fill_value=7.0, dtype=np.float32)


def test_score_and_maps_normalizes_tuple_return():
    x = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(3)]
    scores, maps = score_and_maps(_TupleDetector(), x)
    assert scores.shape == (3,)
    assert maps is not None
    assert maps.shape == (3, 3, 3)


def test_score_and_maps_broadcasts_single_image_map_api():
    x = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(2)]
    scores, maps = score_and_maps(_SingleMapDetector(), x)
    assert scores.shape == (2,)
    assert maps is not None
    assert maps.shape == (2, 3, 3)


def test_score_and_maps_accepts_single_predict_map_2d_output() -> None:
    x = [np.zeros((4, 4, 3), dtype=np.uint8)]
    scores, maps = score_and_maps(_PredictMap2DDetector(), x)
    assert scores.shape == (1,)
    assert maps is not None
    assert maps.shape == (1, 4, 4)
    assert float(maps[0, 0, 0]) == 7.0
