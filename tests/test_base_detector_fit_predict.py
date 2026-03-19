from __future__ import annotations

import numpy as np


def test_base_detector_fit_predict_roundtrip() -> None:
    from pyimgano.models.base_detector import BaseDetector

    class DummyDetector(BaseDetector):
        def fit(self, x, y=None):
            x = np.asarray(x, dtype=np.float64)
            self._set_n_classes(y)
            self.decision_scores_ = np.sum(x, axis=1)
            self._process_decision_scores()
            return self

        def decision_function(self, x):
            x = np.asarray(x, dtype=np.float64)
            return np.sum(x, axis=1)

    x = np.array([[0.0, 0.0], [10.0, 0.0], [1.0, 1.0]])
    det = DummyDetector(contamination=0.34)
    labels = det.fit_predict(x)
    assert labels.shape == (3,)
    assert set(np.unique(labels)).issubset({0, 1})
