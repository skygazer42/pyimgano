from __future__ import annotations

from typing import Sequence

import numpy as np

from pyimgano.defects.roi import roi_mask_from_xyxy_norm


def apply_roi_to_map(
    anomaly_map: np.ndarray,
    roi_xyxy_norm: Sequence[float] | None,
) -> np.ndarray:
    """Zero out anomaly-map pixels outside ROI.

    Args:
        anomaly_map: HxW anomaly map (float-like). Returned as float32.
        roi_xyxy_norm: Optional normalized ROI rectangle in xyxy format.

    Returns:
        Float32 anomaly map with pixels outside ROI set to 0.
    """

    m = np.asarray(anomaly_map, dtype=np.float32)
    if roi_xyxy_norm is None:
        return m

    roi_mask = roi_mask_from_xyxy_norm(m.shape, roi_xyxy_norm).astype(np.float32)
    return m * roi_mask


def apply_border_ignore_to_map(anomaly_map: np.ndarray, *, border_ignore_px: int) -> np.ndarray:
    """Zero out anomaly-map pixels near the border.

    Industrial anomaly maps often contain edge artifacts (resize padding, tiling
    seams, sensor border noise). This helper is an opt-in suppression step for
    defects extraction.

    Args:
        anomaly_map: HxW anomaly map (float-like). Returned as float32.
        border_ignore_px: Number of pixels to zero from each border. 0 disables.

    Returns:
        Float32 anomaly map with border pixels set to 0.
    """

    m = np.asarray(anomaly_map, dtype=np.float32)
    n = int(border_ignore_px)
    if n <= 0:
        return m

    h, w = int(m.shape[0]), int(m.shape[1])
    if h <= 0 or w <= 0:
        return m

    # If the border thickness exceeds the map size, suppress everything.
    if n * 2 >= h or n * 2 >= w:
        return np.zeros_like(m, dtype=np.float32)

    out = m.copy()
    out[:n, :] = 0.0
    out[-n:, :] = 0.0
    out[:, :n] = 0.0
    out[:, -n:] = 0.0
    return out


def compute_roi_stats(
    anomaly_map: np.ndarray,
    roi_xyxy_norm: Sequence[float] | None,
) -> dict[str, float]:
    """Compute simple stats over ROI pixels (max/mean)."""

    m = np.asarray(anomaly_map, dtype=np.float32)
    if roi_xyxy_norm is None:
        values = m.reshape(-1)
    else:
        roi_mask = roi_mask_from_xyxy_norm(m.shape, roi_xyxy_norm).astype(bool)
        values = m[roi_mask]

    if values.size == 0:
        return {"max": 0.0, "mean": 0.0}

    return {"max": float(values.max()), "mean": float(values.mean())}
