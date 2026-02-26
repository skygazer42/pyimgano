import numpy as np
import pytest

from pyimgano.models import create_model


class _DetA:
    def __init__(self):
        self.fit_calls = 0

    def fit(self, X):
        _ = X
        self.fit_calls += 1
        return self

    def decision_function(self, X):
        # 0,1,2,...
        return np.arange(len(X), dtype=np.float32)


class _DetB:
    def __init__(self):
        self.fit_calls = 0

    def fit(self, X):
        _ = X
        self.fit_calls += 1
        return self

    def decision_function(self, X):
        # 10,20,30,...
        return (np.arange(len(X), dtype=np.float32) + 1.0) * 10.0


def test_score_ensemble_mean_rank_combines_scores_scale_invariantly():
    a = _DetA()
    b = _DetB()
    det = create_model("vision_score_ensemble", detectors=[a, b], contamination=0.34)

    X = [0, 1, 2]
    scores = det.decision_function(X)
    assert scores == pytest.approx([0.0, 0.5, 1.0])

    det.fit(X)
    assert a.fit_calls == 1
    assert b.fit_calls == 1
    assert det.threshold_ is not None

    labels = det.predict(X)
    assert labels.shape == (3,)
    assert set(labels.tolist()).issubset({0, 1})


def test_score_ensemble_mean_combines_raw_scores():
    a = _DetA()
    b = _DetB()
    det = create_model("vision_score_ensemble", detectors=[a, b], combine="mean")

    X = [0, 1, 2]
    scores = det.decision_function(X)
    assert scores == pytest.approx([5.0, 10.5, 16.0])


def test_score_ensemble_max_rank_uses_most_anomalous_rank():
    a = _DetA()
    b = _DetB()
    det = create_model("vision_score_ensemble", detectors=[a, b], combine="max_rank")

    X = [0, 1, 2]
    scores = det.decision_function(X)
    assert scores == pytest.approx([0.0, 0.5, 1.0])


def test_score_ensemble_trimmed_mean_drops_extremes():
    class _DetC:
        def fit(self, X):  # noqa: ANN001, ANN201 - test helper
            _ = X
            return self

        def decision_function(self, X):  # noqa: ANN001, ANN201 - test helper
            return (np.arange(len(X), dtype=np.float32) + 1.0) * 100.0

    a = _DetA()
    b = _DetB()
    c = _DetC()
    det = create_model(
        "vision_score_ensemble",
        detectors=[a, b, c],
        combine="trimmed_mean",
        trim_fraction=0.34,
    )

    X = [0, 1, 2]
    scores = det.decision_function(X)
    assert scores == pytest.approx([10.0, 20.0, 30.0])
