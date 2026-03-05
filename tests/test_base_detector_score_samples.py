from __future__ import annotations

import numpy as np


def test_base_detector_score_samples_is_decision_function() -> None:
    from pyimgano.models.base_detector import BaseDetector

    class DummyDetector(BaseDetector):
        def fit(self, X, y=None):
            X = np.asarray(X, dtype=np.float64)
            self._set_n_classes(y)
            self.decision_scores_ = np.sum(X, axis=1)
            self._process_decision_scores()
            return self

        def decision_function(self, X):
            X = np.asarray(X, dtype=np.float64)
            return np.sum(X, axis=1)

    X = np.array([[0.0, 0.0], [10.0, 0.0], [1.0, 1.0]])
    det = DummyDetector().fit(X)
    assert np.allclose(det.score_samples(X), det.decision_function(X))
