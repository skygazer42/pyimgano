import numpy as np
import pytest


def test_predict_proba_linear_returns_two_class_probabilities() -> None:
    from pyimgano.models.base_detector import BaseDetector

    class Dummy(BaseDetector):
        def fit(self, X, y=None):  # noqa: ANN001
            self._set_n_classes(y)
            self.decision_scores_ = np.asarray(X, dtype=np.float64)
            self._process_decision_scores()
            return self

        def decision_function(self, X):  # noqa: ANN001
            return np.asarray(X, dtype=np.float64)

    det = Dummy(contamination=0.2).fit([0, 1, 2, 3, 4])

    proba = np.asarray(det.predict_proba([0, 4], method="linear"), dtype=np.float64)
    assert proba.shape == (2, 2)
    assert np.all(np.isfinite(proba))
    assert np.all(proba >= 0.0)
    assert np.all(proba <= 1.0)
    # rows should sum to 1
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_predict_proba_rejects_unknown_method() -> None:
    from pyimgano.models.base_detector import BaseDetector

    class Dummy(BaseDetector):
        def fit(self, X, y=None):  # noqa: ANN001
            self.decision_scores_ = np.asarray(X, dtype=np.float64)
            self._process_decision_scores()
            return self

        def decision_function(self, X):  # noqa: ANN001
            return np.asarray(X, dtype=np.float64)

    det = Dummy(contamination=0.2).fit([0, 1, 2, 3, 4])

    with pytest.raises(ValueError, match="method"):
        det.predict_proba([0, 4], method="__nope__")
