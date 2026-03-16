from __future__ import annotations

import numpy as np

from pyimgano.models.lscp import (
    CoreLSCP,
    CoreLSCPModel,
    CoreLSCPSpecModel,
    VisionLSCP,
    VisionLSCPSpec,
)


class _DummyDetector:
    """Minimal detector stub to exercise LSCP without heavy dependencies."""

    def fit(self, X):  # noqa: ANN001, ANN201 - sklearn-like API
        x_arr = np.asarray(X)
        self.decision_scores_ = np.arange(x_arr.shape[0], dtype=np.float64)
        return self

    def decision_function(self, X):  # noqa: ANN001, ANN201 - sklearn-like API
        x_arr = np.asarray(X)
        return np.arange(x_arr.shape[0], dtype=np.float64)


def test_lscp_constructors_and_fit_cover_sonar_fixes() -> None:
    d1 = _DummyDetector()
    d2 = _DummyDetector()

    # Cover constructor paths where Sonar suggests dict literal syntax.
    CoreLSCPModel(detector_list=[d1, d2], contamination=0.1)
    CoreLSCPSpecModel(detector_specs=[d1, d2], contamination=0.1)
    VisionLSCP(feature_extractor="identity", detector_list=[d1, d2], contamination=0.1)
    VisionLSCPSpec(feature_extractor="identity", detector_specs=[d1, d2], contamination=0.1)

    # Cover CoreLSCP.fit(...) handling of the sklearn-style `y` argument.
    X = np.arange(20, dtype=np.float64).reshape(10, 2)
    det = CoreLSCP(
        detector_list=[_DummyDetector(), _DummyDetector()],
        local_region_iterations=1,
        local_region_size=3,
        random_state=0,
    )
    det.fit(X, y=np.zeros((X.shape[0],), dtype=np.int64))
    scores = det.decision_function(X)
    assert scores.shape == (X.shape[0],)
