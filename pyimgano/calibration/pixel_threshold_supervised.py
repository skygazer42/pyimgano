from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _segf1_from_counts(*, tp: int, fp: int, fn: int) -> float:
    tp_i = int(tp)
    fp_i = int(fp)
    fn_i = int(fn)

    if tp_i < 0 or fp_i < 0 or fn_i < 0:
        raise ValueError("tp/fp/fn must be non-negative")

    # If there are no positives in GT, define F1=1 only when we predict none.
    if (tp_i + fn_i) == 0:
        return 1.0 if (tp_i + fp_i) == 0 else 0.0

    precision = tp_i / (tp_i + fp_i) if (tp_i + fp_i) > 0 else 0.0
    recall = tp_i / (tp_i + fn_i) if (tp_i + fn_i) > 0 else 0.0
    denom = precision + recall
    if denom <= 0.0:
        return 0.0
    return float(2.0 * precision * recall / denom)


def calibrate_supervised_segf1_threshold(
    pixel_scores: NDArray,
    pixel_labels: NDArray,
    *,
    max_candidates: int = 512,
) -> float:
    """Calibrate a pixel threshold by maximizing SegF1 on labeled pixels.

    Parameters
    ----------
    pixel_scores:
        Pixel anomaly scores. Shape can be (N,H,W) or (H,W).
    pixel_labels:
        Pixel GT labels/masks aligned with ``pixel_scores``. Non-zero is treated
        as anomaly/foreground.
    max_candidates:
        Upper bound on the number of thresholds to evaluate (best-effort). When
        the score field has too many unique values, this falls back to a fixed
        quantile grid.

    Returns
    -------
    threshold:
        A float threshold suitable for binarizing pixel anomaly maps via
        ``pixel_scores >= threshold``.

    Notes
    -----
    If multiple thresholds achieve the same best F1, we pick the *highest*
    threshold (more conservative; fewer false positives).
    """

    scores = np.asarray(pixel_scores, dtype=np.float64)
    labels = np.asarray(pixel_labels)

    if scores.size == 0:
        raise ValueError("pixel_scores must be non-empty.")
    if labels.shape != scores.shape:
        raise ValueError(
            "pixel_labels and pixel_scores must have the same shape. "
            f"Got labels={labels.shape} vs scores={scores.shape}."
        )

    s = scores.reshape(-1)
    y = labels.reshape(-1) > 0

    if not np.all(np.isfinite(s)):
        s = np.nan_to_num(s, nan=-np.inf, posinf=np.inf, neginf=-np.inf)

    unique = np.unique(s)
    max_c = int(max_candidates)
    if max_c < 2:
        max_c = 2
    if unique.size > max_c:
        qs = np.linspace(0.0, 1.0, num=max_c, dtype=np.float64)
        unique = np.unique(np.quantile(s, qs))

    best_f1 = -1.0
    best_thr = float(unique[0])

    for thr in unique:
        pred = s >= float(thr)
        tp = int(np.sum(pred & y))
        fp = int(np.sum(pred & ~y))
        fn = int(np.sum((~pred) & y))
        f1 = _segf1_from_counts(tp=tp, fp=fp, fn=fn)

        if f1 > best_f1 + 1e-12:
            best_f1 = float(f1)
            best_thr = float(thr)
        elif abs(f1 - best_f1) <= 1e-12 and float(thr) > float(best_thr):
            best_thr = float(thr)

    return float(best_thr)
