"""Small uint8 image validation helpers.

Several preprocessing/synthesis utilities operate strictly on uint8 images and
repeat the same dtype/shape checks. Centralize those checks to:
- reduce SonarCloud duplicated code on new changes
- keep behavior consistent across utilities
"""

from __future__ import annotations

import numpy as np


def as_u8_image(image: np.ndarray) -> np.ndarray:
    """Validate `image` is a uint8 grayscale (H,W) or color (H,W,3) array.

    Returns the input as a numpy array view when possible.
    """

    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        raise ValueError(f"Expected uint8 image, got dtype={arr.dtype}")
    if arr.ndim not in (2, 3):
        raise ValueError(f"Expected grayscale (H,W) or color (H,W,3) image, got {arr.shape}")
    if arr.ndim == 3 and arr.shape[2] != 3:
        raise ValueError(f"Expected grayscale (H,W) or color (H,W,3) image, got {arr.shape}")
    return arr
