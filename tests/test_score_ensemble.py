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
