import numpy as np


def test_base_detector_can_use_pot_thresholding() -> None:
    from pyimgano.models.base_detector import BaseDetector

    class Dummy(BaseDetector):
        def fit(self, X, y=None):  # noqa: ANN001
            self._set_n_classes(y)
            self.decision_scores_ = np.asarray(X, dtype=np.float64)
            self._process_decision_scores()
            return self

        def decision_function(self, X):  # noqa: ANN001
            return np.asarray(X, dtype=np.float64)

    rng = np.random.default_rng(0)
    train_scores = rng.exponential(scale=1.0, size=500)

    det = Dummy(contamination=0.1).use_pot_thresholding(tail_fraction=0.2, min_exceedances=20)
    det.fit(train_scores)

    assert hasattr(det, "threshold_")
    assert np.isfinite(det.threshold_)
    assert hasattr(det, "labels_")
    labels = np.asarray(det.labels_, dtype=int)
    assert labels.shape == (500,)
    assert set(np.unique(labels)).issubset({0, 1})
    assert hasattr(det, "pot_info_")
