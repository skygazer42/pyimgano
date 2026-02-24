from __future__ import annotations

from typing import Sequence

import numpy as np


def clamp_roi_xyxy_norm(roi_xyxy_norm: Sequence[float]) -> list[float]:
    """Clamp and normalize a rectangle ROI in xyxy normalized coordinates.

    Args:
        roi_xyxy_norm: Sequence of 4 values: [x1, y1, x2, y2] in normalized space.

    Returns:
        A list [x1, y1, x2, y2] with:
        - values clamped to [0, 1]
        - ordered so x1<=x2 and y1<=y2
    """

    if len(roi_xyxy_norm) != 4:
        raise ValueError(f"roi_xyxy_norm must have length 4, got {len(roi_xyxy_norm)}")

    x1, y1, x2, y2 = (float(v) for v in roi_xyxy_norm)

    def _clamp01(v: float) -> float:
        return float(min(max(v, 0.0), 1.0))

    x1c, y1c, x2c, y2c = (_clamp01(x1), _clamp01(y1), _clamp01(x2), _clamp01(y2))
    return [min(x1c, x2c), min(y1c, y2c), max(x1c, x2c), max(y1c, y2c)]


def roi_mask_from_xyxy_norm(
    shape_hw: tuple[int, int],
    roi_xyxy_norm: Sequence[float],
) -> np.ndarray:
    """Create a uint8 ROI mask (1 inside ROI, 0 outside) for an HxW grid."""

    h, w = int(shape_hw[0]), int(shape_hw[1])
    if h <= 0 or w <= 0:
        raise ValueError(f"shape_hw must be positive, got {(h, w)}")

    x1, y1, x2, y2 = clamp_roi_xyxy_norm(roi_xyxy_norm)

    x_start = int(np.floor(x1 * w))
    x_end = int(np.ceil(x2 * w))
    y_start = int(np.floor(y1 * h))
    y_end = int(np.ceil(y2 * h))

    x_start = max(min(x_start, w), 0)
    x_end = max(min(x_end, w), 0)
    y_start = max(min(y_start, h), 0)
    y_end = max(min(y_end, h), 0)

    mask = np.zeros((h, w), dtype=np.uint8)
    if x_end > x_start and y_end > y_start:
        mask[y_start:y_end, x_start:x_end] = 1
    return mask

