import numpy as np

from pyimgano.calibration.fewshot import apply_threshold, fit_quantile_threshold, fit_threshold


def test_fit_threshold_separates():
    normal = np.array([0.1, 0.2, 0.3])
    anomaly = np.array([0.9, 0.8])
    thr = fit_threshold(normal, anomaly)
    assert 0.3 <= thr <= 0.9


def test_fit_quantile_threshold():
    normal = np.array([0.0, 1.0, 2.0, 3.0])
    thr = fit_quantile_threshold(normal, contamination=0.25)
    assert 2.0 <= thr <= 3.0


def test_apply_threshold():
    scores = np.array([0.1, 0.5, 0.9])
    labels = apply_threshold(scores, threshold=0.5)
    assert labels.tolist() == [0, 1, 1]

