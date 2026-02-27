from __future__ import annotations

import numpy as np


def test_core_feature_detector_contract_fit_predict() -> None:
    from pyimgano.models.core_feature_base import CoreFeatureDetector

    class _Backend:
        def fit(self, X):  # noqa: ANN001, ANN201 - test helper
            self.decision_scores_ = np.asarray(X).sum(axis=1)
            return self

        def decision_function(self, X):  # noqa: ANN001, ANN201 - test helper
            return np.asarray(X).sum(axis=1)

    class _Demo(CoreFeatureDetector):
        def __init__(self, *, contamination: float = 0.1) -> None:
            super().__init__(contamination=contamination)

        def _build_detector(self):  # noqa: ANN201
            return _Backend()

    X = np.arange(30, dtype=np.float32).reshape(10, 3)
    det = _Demo(contamination=0.2)
    det.fit(X)

    assert hasattr(det, "decision_scores_")
    assert hasattr(det, "threshold_")
    assert hasattr(det, "labels_")
    assert det.decision_scores_.shape == (X.shape[0],)
    assert det.labels_.shape == (X.shape[0],)

    scores = det.decision_function(X)
    labels = det.predict(X)
    assert scores.shape == (X.shape[0],)
    assert labels.shape == (X.shape[0],)
    assert set(labels).issubset({0, 1})

