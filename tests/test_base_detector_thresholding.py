import numpy as np
import pytest


def test_base_detector_validates_contamination_range() -> None:
    from pyimgano.models.base_detector import BaseDetector

    with pytest.raises(ValueError, match="contamination"):
        BaseDetector(contamination=0.0)

    with pytest.raises(ValueError, match="contamination"):
        BaseDetector(contamination=-0.1)

    with pytest.raises(ValueError, match="contamination"):
        BaseDetector(contamination=0.9)

    # Boundary: PyOD allows 0.5
    BaseDetector(contamination=0.5)


def test_process_decision_scores_sets_threshold_and_labels() -> None:
    from pyimgano.models.base_detector import BaseDetector

    class Dummy(BaseDetector):
        def fit(self, X, y=None):  # noqa: ANN001
            self.decision_scores_ = np.asarray(X, dtype=np.float64)
            self._process_decision_scores()
            return self

        def decision_function(self, X):  # noqa: ANN001
            return np.asarray(X, dtype=np.float64)

    det = Dummy(contamination=0.2)
    det.fit([0, 1, 2, 3, 4])

    assert hasattr(det, "threshold_")
    assert hasattr(det, "labels_")

    labels = np.asarray(det.labels_, dtype=int)
    assert labels.shape == (5,)
    assert set(np.unique(labels)).issubset({0, 1})
    # 20% of 5 samples => 1 outlier
    assert int(labels.sum()) == 1


def test_predict_uses_threshold_and_returns_binary_labels() -> None:
    from pyimgano.models.base_detector import BaseDetector

    class Dummy(BaseDetector):
        def fit(self, X, y=None):  # noqa: ANN001
            self.decision_scores_ = np.asarray(X, dtype=np.float64)
            self._process_decision_scores()
            return self

        def decision_function(self, X):  # noqa: ANN001
            return np.asarray(X, dtype=np.float64)

    det = Dummy(contamination=0.2).fit([0, 1, 2, 3, 4])

    preds = np.asarray(det.predict([0, 4]), dtype=int)
    assert preds.shape == (2,)
    assert set(np.unique(preds)).issubset({0, 1})
    assert preds.tolist() == [0, 1]
