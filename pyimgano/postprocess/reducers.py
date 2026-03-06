from __future__ import annotations

"""Pixel-map to image-score reducers.

Many industrial detectors produce an anomaly heatmap (H,W). Downstream systems
often need a single image-level score for ranking / thresholding.
"""

import math
from typing import Literal

import numpy as np

Reducer = Literal["max", "mean", "topk_mean", "area"]


def reduce_anomaly_map(
    anomaly_map: np.ndarray,
    *,
    method: Reducer = "topk_mean",
    topk: float = 0.01,
    area_threshold: float = 0.5,
) -> float:
    """Reduce a 2D anomaly map into one scalar score.

    Parameters
    ----------
    method:
        - `max`: maximum pixel score
        - `mean`: average pixel score
        - `topk_mean`: mean of top-k fraction pixels
        - `area`: fraction of pixels above `area_threshold`
    """

    m = np.asarray(anomaly_map, dtype=np.float64)
    if m.ndim != 2:
        raise ValueError(f"Expected anomaly_map shape (H,W), got {m.shape}")
    flat = m.reshape(-1)
    if flat.size == 0:
        return 0.0

    method_l = str(method).lower().strip()
    if method_l == "max":
        return float(np.max(flat))
    if method_l == "mean":
        return float(np.mean(flat))
    if method_l == "topk_mean":
        topk_f = float(topk)
        if not (0.0 < topk_f <= 1.0):
            raise ValueError("topk must be in (0,1]")
        k = max(1, int(math.ceil(topk_f * flat.size)))
        k = min(k, flat.size)
        top_vals = np.partition(flat, -k)[-k:]
        return float(np.mean(top_vals))
    if method_l == "area":
        thr = float(area_threshold)
        return float(np.mean(flat >= thr))

    raise ValueError(f"Unknown reducer method: {method}")
