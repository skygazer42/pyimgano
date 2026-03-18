import numpy as np
import pytest


class _DummyConfidenceDetector:
    pass


def _make_dummy_detector():
    from pyimgano.models.base_detector import BaseDetector

    class Dummy(BaseDetector):
        def fit(self, X, y=None):  # noqa: ANN001
            self._set_n_classes(y)
            self.decision_scores_ = np.asarray(X, dtype=np.float64)
            self._process_decision_scores()
            return self

        def decision_function(self, X):  # noqa: ANN001
            return np.asarray(X, dtype=np.float64)

    return Dummy(contamination=0.2).fit([0, 1, 2, 3, 4])


def test_predict_confidence_returns_normalized_label_confidence() -> None:
    det = _make_dummy_detector()

    conf = np.asarray(det.predict_confidence([0, 2, 4]), dtype=np.float64)

    assert conf.shape == (3,)
    assert np.all(np.isfinite(conf))
    assert np.all(conf >= 0.0)
    assert np.all(conf <= 1.0)
    assert conf[0] > conf[1]
    assert conf[2] > conf[1]


def test_predict_return_confidence_returns_labels_and_confidence() -> None:
    det = _make_dummy_detector()

    labels, conf = det.predict([0, 2, 4], return_confidence=True)

    labels_arr = np.asarray(labels, dtype=np.int64)
    conf_arr = np.asarray(conf, dtype=np.float64)
    assert labels_arr.tolist() == [0, 0, 1]
    assert conf_arr.shape == (3,)
    assert conf_arr[0] > conf_arr[1]
    assert conf_arr[2] > conf_arr[1]


def test_predict_with_rejection_rejects_low_confidence_samples() -> None:
    det = _make_dummy_detector()

    labels, conf = det.predict_with_rejection([0, 2, 4], confidence_threshold=0.75, return_confidence=True)

    labels_arr = np.asarray(labels, dtype=np.int64)
    conf_arr = np.asarray(conf, dtype=np.float64)
    assert labels_arr.tolist() == [0, -2, 1]
    assert conf_arr.shape == (3,)
    assert conf_arr[1] < 0.75


def test_predict_with_rejection_validates_threshold() -> None:
    det = _make_dummy_detector()

    with pytest.raises(ValueError, match="confidence_threshold"):
        det.predict_with_rejection([0, 2, 4], confidence_threshold=0.0)
