from __future__ import annotations

import cv2
import numpy as np


def smooth_anomaly_map(
    anomaly_map: np.ndarray,
    *,
    method: str,
    ksize: int,
    sigma: float = 0.0,
) -> np.ndarray:
    """Optionally smooth an anomaly map before thresholding.

    This is intended for industrial defects extraction to reduce spurious
    single-pixel false positives.

    Args:
        anomaly_map: 2D HxW anomaly map (float-like). Returned as float32.
        method: "none" | "median" | "gaussian" | "box"
        ksize: Kernel size. For median/box it should be >=3. For gaussian it can
            be 0 when sigma>0 (OpenCV chooses a size).
        sigma: Gaussian sigma (only used for method="gaussian"). 0 means
            OpenCV default.

    Returns:
        Smoothed float32 anomaly map (same shape).
    """

    m = np.asarray(anomaly_map, dtype=np.float32)
    if m.ndim != 2:
        raise ValueError(f"anomaly_map must be 2D (H, W), got shape {m.shape}")

    kind = str(method).lower().strip()
    if kind in ("", "none", "off", "false", "0"):
        return m

    k = int(ksize)

    if kind == "median":
        if k <= 1:
            return m
        if k % 2 == 0:
            k += 1
        return np.asarray(cv2.medianBlur(m, k), dtype=np.float32)

    if kind == "gaussian":
        s = float(sigma)
        if k <= 0:
            if s <= 0:
                return m
            return np.asarray(cv2.GaussianBlur(m, ksize=(0, 0), sigmaX=s), dtype=np.float32)
        if k % 2 == 0:
            k += 1
        return np.asarray(cv2.GaussianBlur(m, ksize=(k, k), sigmaX=s), dtype=np.float32)

    if kind == "box":
        if k <= 1:
            return m
        return np.asarray(cv2.blur(m, ksize=(k, k)), dtype=np.float32)

    raise ValueError(f"Unknown smoothing method: {method!r}. Expected none|median|gaussian|box.")

