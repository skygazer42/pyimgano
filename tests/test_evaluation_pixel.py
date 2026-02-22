import numpy as np

from pyimgano.calibration.pixel_threshold import calibrate_normal_pixel_quantile_threshold
from pyimgano.evaluation import (
    compute_aupro,
    compute_bg_fpr,
    compute_pixel_auroc,
    compute_pixel_average_precision,
    compute_pixel_segf1,
)


def test_pixel_metrics_perfect():
    pixel_labels = np.zeros((1, 10, 10), dtype=np.uint8)
    pixel_labels[:, 2:5, 2:5] = 1

    pixel_scores = pixel_labels.astype(np.float32)

    assert compute_pixel_auroc(pixel_labels, pixel_scores) > 0.99
    assert compute_pixel_average_precision(pixel_labels, pixel_scores) > 0.99
    assert 0.0 <= compute_aupro(pixel_labels, pixel_scores) <= 1.0


def test_pixel_segf1_and_bg_fpr_single_threshold_perfect() -> None:
    pixel_labels = np.array([[[0, 1], [0, 1]]], dtype=np.uint8)
    pixel_scores = np.array([[[0.1, 0.9], [0.2, 0.8]]], dtype=np.float32)
    thr = 0.5

    assert compute_pixel_segf1(pixel_labels, pixel_scores, threshold=thr) == 1.0
    assert compute_bg_fpr(pixel_labels, pixel_scores, threshold=thr) == 0.0


def test_bg_fpr_counts_only_background_pixels() -> None:
    pixel_labels = np.array([[[0, 1], [0, 1]]], dtype=np.uint8)
    pixel_scores = np.array([[[0.7, 0.9], [0.2, 0.8]]], dtype=np.float32)
    thr = 0.5

    assert compute_bg_fpr(pixel_labels, pixel_scores, threshold=thr) == 0.5


def test_bg_fpr_uses_strict_threshold() -> None:
    labels = np.zeros((1, 1, 1), dtype=np.uint8)
    scores = np.zeros((1, 1, 1), dtype=np.float32)

    # If we calibrate a threshold at exactly the max normal score (common for quantiles),
    # strict `>` avoids flagging those pixels as false-positives.
    assert compute_bg_fpr(labels, scores, threshold=0.0) == 0.0


def test_calibrate_normal_pixel_quantile_threshold_uses_background_pixels_only() -> None:
    labels = np.array([[[0, 1], [0, 0]]], dtype=np.uint8)
    scores = np.array([[[0.1, 100.0], [0.2, 0.3]]], dtype=np.float32)

    # Background scores are [0.1, 0.2, 0.3] so q=1.0 should pick the max background value.
    thr = calibrate_normal_pixel_quantile_threshold(scores, pixel_labels=labels, q=1.0)
    assert np.isclose(thr, 0.3, atol=1e-6)
