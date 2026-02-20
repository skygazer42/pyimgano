import numpy as np

from pyimgano.evaluation import (
    compute_aupro,
    compute_pixel_auroc,
    compute_pixel_average_precision,
)


def test_pixel_metrics_perfect():
    pixel_labels = np.zeros((1, 10, 10), dtype=np.uint8)
    pixel_labels[:, 2:5, 2:5] = 1

    pixel_scores = pixel_labels.astype(np.float32)

    assert compute_pixel_auroc(pixel_labels, pixel_scores) > 0.99
    assert compute_pixel_average_precision(pixel_labels, pixel_scores) > 0.99
    assert 0.0 <= compute_aupro(pixel_labels, pixel_scores) <= 1.0

