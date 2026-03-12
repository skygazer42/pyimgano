from __future__ import annotations

import numpy as np

from pyimgano.models.base_detector import BaseDetector
from pyimgano.models.protocols import (
    DetectorProtocol,
    PixelMapDetectorProtocol,
    normalize_anomaly_maps,
)


class _DummyDetector(BaseDetector):
    input_mode = "numpy"

    def fit(self, X, y=None):
        self.decision_scores_ = np.asarray([0.1, 0.9], dtype=np.float64)
        self._process_decision_scores()
        return self

    def decision_function(self, X):
        n = len(list(X))
        return np.zeros((n,), dtype=np.float32)


class _DummyPixelDetector(_DummyDetector):
    def predict_anomaly_map(self, X):
        n = len(list(X))
        return [np.ones((4, 4), dtype=np.float32) for _ in range(n)]


def test_detector_runtime_protocol_accepts_native_detector():
    detector = _DummyDetector()
    assert isinstance(detector, DetectorProtocol)
    assert detector.input_mode == "numpy"


def test_normalize_anomaly_maps_accepts_list_outputs():
    detector = _DummyPixelDetector()
    assert isinstance(detector, PixelMapDetectorProtocol)
    maps = normalize_anomaly_maps(detector.predict_anomaly_map([0, 1]), n_expected=2)
    assert maps.shape == (2, 4, 4)


def test_normalize_anomaly_maps_accepts_tuple_outputs():
    maps = normalize_anomaly_maps(
        (
            np.ones((4, 4), dtype=np.float32),
            np.zeros((4, 4), dtype=np.float32),
        ),
        n_expected=2,
    )
    assert maps.shape == (2, 4, 4)
