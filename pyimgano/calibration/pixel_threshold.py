from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray


def calibrate_normal_pixel_quantile_threshold(
    pixel_scores: NDArray,
    *,
    pixel_labels: Optional[NDArray] = None,
    q: float = 0.999,
) -> float:
    """Calibrate a pixel threshold using a quantile of *normal* pixels.

    Parameters
    ----------
    pixel_scores:
        Pixel anomaly scores. Shape can be (N,H,W) or (H,W).
    pixel_labels:
        Optional GT mask aligned with ``pixel_scores``. If provided, only pixels
        where ``pixel_labels==0`` are used for calibration.
    q:
        Quantile in [0,1]. Typical industrial defaults are high (e.g. 0.999).

    Returns
    -------
    threshold:
        A float threshold suitable for binarizing pixel anomaly maps via
        ``pixel_scores >= threshold``.
    """

    qf = float(q)
    if not (0.0 <= qf <= 1.0):
        raise ValueError(f"q must be in [0,1], got {q}")

    scores = np.asarray(pixel_scores, dtype=np.float64)
    if scores.size == 0:
        raise ValueError("pixel_scores must be non-empty.")

    if pixel_labels is not None:
        labels = np.asarray(pixel_labels)
        if labels.shape != scores.shape:
            raise ValueError(
                "pixel_labels and pixel_scores must have the same shape. "
                f"Got labels={labels.shape} vs scores={scores.shape}."
            )
        bg = (labels <= 0)
        if int(np.sum(bg)) == 0:
            raise ValueError("No background (normal) pixels found in pixel_labels.")
        scores = scores[bg]

    if not np.all(np.isfinite(scores)):
        scores = np.nan_to_num(scores, nan=-np.inf, posinf=np.inf, neginf=-np.inf)

    thr = float(np.quantile(scores.reshape(-1), qf))
    return thr

