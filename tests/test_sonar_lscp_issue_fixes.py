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

    def fit(self, x):  # noqa: ANN001, ANN201 - sklearn-like API
        x_arr = np.asarray(x)
        self.decision_scores_ = np.arange(x_arr.shape[0], dtype=np.float64)
        return self

    def decision_function(self, x):  # noqa: ANN001, ANN201 - sklearn-like API
        x_arr = np.asarray(x)
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
    x = np.arange(20, dtype=np.float64).reshape(10, 2)
    det = CoreLSCP(
        detector_list=[_DummyDetector(), _DummyDetector()],
        local_region_iterations=1,
        local_region_size=3,
        random_state=0,
    )
    det.fit(x, y=np.zeros((x.shape[0],), dtype=np.int64))
    scores = det.decision_function(x)
    assert scores.shape == (x.shape[0],)


def test_lscp_accepts_numpy_generator_random_state() -> None:
    x = np.arange(24, dtype=np.float64).reshape(12, 2)
    det = CoreLSCP(
        detector_list=[_DummyDetector(), _DummyDetector()],
        local_region_iterations=1,
        local_region_size=4,
        random_state=np.random.default_rng(0),
    )

    det.fit(x)
    scores = det.decision_function(x[:4])

    assert scores.shape == (4,)


def test_lscp_competent_detector_selection_handles_nearly_constant_scores() -> None:
    det = CoreLSCP(detector_list=[object(), object(), object(), object(), object()], n_bins=3)

    scores = np.asarray(
        [1.0, 1.0, 1.0, 1.0, np.nextafter(1.0, 2.0)],
        dtype=np.float64,
    )

    idx = det._get_competent_detectors(scores)

    assert np.array_equal(idx, np.asarray([4], dtype=np.int64))
