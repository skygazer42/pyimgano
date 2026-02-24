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

