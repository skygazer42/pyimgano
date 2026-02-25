from __future__ import annotations

import cv2
import numpy as np


def hysteresis_anomaly_map_to_binary_mask(
    anomaly_map: np.ndarray,
    *,
    low: float,
    high: float,
) -> np.ndarray:
    """Hysteresis thresholding for anomaly maps (returns uint8 0/255 mask).

    This keeps only the low-threshold connected components that contain at least
    one high-threshold "seed" pixel. It is useful for reducing speckle while
    keeping defect regions connected.
    """

    m = np.asarray(anomaly_map, dtype=np.float32)
    if m.ndim != 2:
        raise ValueError(f"anomaly_map must be 2D (H, W), got shape {m.shape}")

    lo = float(low)
    hi = float(high)
    if hi < lo:
        lo, hi = hi, lo

    low_bin = (m >= lo).astype(np.uint8)
    high_bin = (m >= hi).astype(np.uint8)

    if int(high_bin.max()) == 0:
        return np.zeros_like(low_bin, dtype=np.uint8)

    num_labels, labels, _stats, _centroids = cv2.connectedComponentsWithStats(low_bin, connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(low_bin, dtype=np.uint8)

    seed_labels = np.unique(labels[high_bin > 0])
    seed_labels = seed_labels[seed_labels != 0]
    if seed_labels.size == 0:
        return np.zeros_like(low_bin, dtype=np.uint8)

    keep = np.isin(labels, seed_labels).astype(np.uint8)
    return keep * 255

