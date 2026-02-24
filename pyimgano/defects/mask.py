from __future__ import annotations

import numpy as np


def anomaly_map_to_binary_mask(anomaly_map: np.ndarray, pixel_threshold: float) -> np.ndarray:
    """Convert an anomaly map to a binary uint8 mask (0 or 255).

    Args:
        anomaly_map: HxW anomaly map (float-like).
        pixel_threshold: Pixels >= threshold are considered defective.

    Returns:
        A uint8 mask with values in {0, 255}.
    """

    m = np.asarray(anomaly_map, dtype=np.float32)
    t = float(pixel_threshold)
    return (m >= t).astype(np.uint8) * 255

