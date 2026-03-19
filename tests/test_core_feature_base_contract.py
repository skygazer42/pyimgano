from __future__ import annotations

import numpy as np


def test_core_feature_detector_contract_fit_predict() -> None:
    from pyimgano.models.core_feature_base import CoreFeatureDetector

    class _Backend:
        def fit(self, x):  # noqa: ANN001, ANN201 - test helper
            self.decision_scores_ = np.asarray(x).sum(axis=1)
            return self

        def decision_function(self, x):  # noqa: ANN001, ANN201 - test helper
            return np.asarray(x).sum(axis=1)

    class _Demo(CoreFeatureDetector):
        def __init__(self, *, contamination: float = 0.1) -> None:
            super().__init__(contamination=contamination)

        def _build_detector(self):  # noqa: ANN201
            return _Backend()

    x = np.arange(30, dtype=np.float32).reshape(10, 3)
    det = _Demo(contamination=0.2)
    det.fit(x)

    assert hasattr(det, "decision_scores_")
    assert hasattr(det, "threshold_")
    assert hasattr(det, "labels_")
    assert det.decision_scores_.shape == (x.shape[0],)
    assert det.labels_.shape == (x.shape[0],)

    scores = det.decision_function(x)
    labels = det.predict(x)
    assert scores.shape == (x.shape[0],)
    assert labels.shape == (x.shape[0],)
    assert set(labels).issubset({0, 1})
