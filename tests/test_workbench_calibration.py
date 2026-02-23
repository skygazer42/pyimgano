import numpy as np

from pyimgano.workbench.calibration import calibrate_detector_threshold


def test_calibrate_detector_threshold_uses_contamination_default():
    class _Det:
        contamination = 0.1

        def decision_function(self, X):  # noqa: ANN001
            n = len(list(X))
            return np.arange(n, dtype=np.float32)

    det = _Det()
    inputs = [f"img_{i}.png" for i in range(10)]
    threshold = calibrate_detector_threshold(det, inputs)
    expected = float(np.quantile(np.arange(10, dtype=np.float32), 0.9))
    assert np.isclose(threshold, expected)
    assert np.isclose(getattr(det, "threshold_"), expected)


def test_calibrate_detector_threshold_respects_explicit_quantile():
    class _Det:
        def decision_function(self, X):  # noqa: ANN001
            n = len(list(X))
            return np.arange(n, dtype=np.float32)

    det = _Det()
    inputs = [f"img_{i}.png" for i in range(10)]
    threshold = calibrate_detector_threshold(det, inputs, quantile=0.5)
    expected = float(np.quantile(np.arange(10, dtype=np.float32), 0.5))
    assert np.isclose(threshold, expected)
