from __future__ import annotations

import numpy as np

from pyimgano.inference.runtime_adapter import score_and_maps


class _TupleDetector:
    input_mode = "numpy"

    def decision_function(self, X):
        n = len(list(X))
        scores = np.arange(float(n), dtype=np.float32)
        maps = [np.full((3, 3), fill_value=i, dtype=np.float32) for i in range(n)]
        return scores, maps


class _SingleMapDetector:
    input_mode = "numpy"

    def decision_function(self, X):
        n = len(list(X))
        return np.linspace(0.0, 1.0, num=n, dtype=np.float32)

    def get_anomaly_map(self, X):
        return np.ones((3, 3), dtype=np.float32)


def test_score_and_maps_normalizes_tuple_return():
    X = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(3)]
    scores, maps = score_and_maps(_TupleDetector(), X)
    assert scores.shape == (3,)
    assert maps is not None
    assert maps.shape == (3, 3, 3)


def test_score_and_maps_broadcasts_single_image_map_api():
    X = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(2)]
    scores, maps = score_and_maps(_SingleMapDetector(), X)
    assert scores.shape == (2,)
    assert maps is not None
    assert maps.shape == (2, 3, 3)
