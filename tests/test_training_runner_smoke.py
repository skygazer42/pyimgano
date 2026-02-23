from __future__ import annotations

from pyimgano.training.runner import micro_finetune


def test_micro_finetune_passes_supported_fit_kwargs():
    class _Detector:
        def __init__(self):
            self.called = False
            self.received = None

        def fit(self, X, *, epochs=None, lr=None):  # noqa: ANN001 - test stub
            self.called = True
            self.received = {"epochs": epochs, "lr": lr, "n": len(list(X))}
            return self

    det = _Detector()
    out = micro_finetune(det, ["a", "b"], seed=123, fit_kwargs={"epochs": 2, "lr": 1e-3})

    assert det.called is True
    assert det.received == {"epochs": 2, "lr": 1e-3, "n": 2}
    assert out["fit_kwargs_used"] == {"epochs": 2, "lr": 1e-3}
    assert out["timing"]["fit_s"] >= 0.0
    assert out["timing"]["total_s"] >= out["timing"]["fit_s"]


def test_micro_finetune_falls_back_when_kwargs_unsupported():
    class _Detector:
        def __init__(self):
            self.called = False

        def fit(self, X):  # noqa: ANN001 - test stub
            self.called = True
            self.n = len(list(X))
            return self

    det = _Detector()
    out = micro_finetune(det, ["a"], fit_kwargs={"epochs": 2})

    assert det.called is True
    assert det.n == 1
    assert out["fit_kwargs_used"] == {}

