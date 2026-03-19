from __future__ import annotations

import math

import pytest


def test_base_detector_get_set_params_roundtrip() -> None:
    from pyimgano.models.base_detector import BaseDetector

    class DummyDetector(BaseDetector):
        def __init__(self, contamination: float = 0.1, alpha: float = 1.0) -> None:
            super().__init__(contamination=contamination)
            self.alpha = float(alpha)

        def fit(self, x, y=None):
            del x
            self._set_n_classes(y)
            self.decision_scores_ = [0.0]
            self._process_decision_scores()
            return self

        def decision_function(self, x):
            del x
            return [0.0]

    det = DummyDetector(contamination=0.2, alpha=3.0)
    params = det.get_params()
    assert math.isclose(params["contamination"], 0.2)
    assert math.isclose(params["alpha"], 3.0)

    det.set_params(alpha=5.0)
    assert det.alpha == pytest.approx(5.0)


def test_base_detector_set_params_rejects_unknown() -> None:
    from pyimgano.models.base_detector import BaseDetector

    class DummyDetector(BaseDetector):
        def fit(self, x, y=None):
            del x
            self._set_n_classes(y)
            self.decision_scores_ = [0.0]
            self._process_decision_scores()
            return self

        def decision_function(self, x):
            del x
            return [0.0]

    det = DummyDetector()
    try:
        det.set_params(does_not_exist=123)
    except ValueError as exc:
        assert "does_not_exist" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown parameter")
