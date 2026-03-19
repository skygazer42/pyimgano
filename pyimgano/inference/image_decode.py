from __future__ import annotations

"""Small, dependency-light helpers for decoding images into canonical numpy arrays.

These utilities are intentionally limited and are primarily used by inference-time
wrappers (tiling, preprocessing) that must decode *paths* into numpy arrays.

Core contract:
- output is canonical ``RGB / uint8 / HWC`` (contiguous)
- decoding uses OpenCV ``IMREAD_UNCHANGED`` to preserve 16-bit inputs
- 16-bit inputs are scaled to uint8 via `u16_max` (defaults to 65535 in
  `normalize_numpy_image`)
"""

from pathlib import Path
from typing import Union

import numpy as np
from numpy.typing import NDArray

from pyimgano.inputs.image_format import ImageFormat, normalize_numpy_image
from pyimgano.utils.optional_deps import require


def load_path_as_rgb_u8_hwc(
    path: Union[str, Path],
    *,
    u16_max: int | None = None,
) -> NDArray[np.uint8]:
    """Load an image path into canonical ``RGB/u8/HWC``.

    Notes
    -----
    - OpenCV loads 3-channel images as BGR; we declare that explicitly and let
      `normalize_numpy_image` handle BGR->RGB + dtype scaling.
    - 4-channel inputs (BGRA) are supported by dropping alpha.
    """

    cv2 = require("cv2", purpose="loading images from disk for inference wrappers")

    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to load image: {path}")

    arr = np.asarray(img)

    fmt: ImageFormat
    if arr.ndim == 2:
        if arr.dtype == np.uint8:
            fmt = ImageFormat.GRAY_U8_HW
        elif arr.dtype == np.uint16:
            fmt = ImageFormat.GRAY_U16_HW
        else:
            raise ValueError(
                f"Unsupported grayscale dtype for path input. Got dtype={arr.dtype} for {path}."
            )
        out = normalize_numpy_image(arr, input_format=fmt, u16_max=u16_max)
        return np.ascontiguousarray(out, dtype=np.uint8)

    if arr.ndim == 3:
        ch = int(arr.shape[2])

        # Handle common alpha variants.
        if ch == 4:
            arr = arr[:, :, :3]
            ch = 3

        if ch == 1:
            if arr.dtype == np.uint8:
                fmt = ImageFormat.GRAY_U8_HWC1
            elif arr.dtype == np.uint16:
                fmt = ImageFormat.GRAY_U16_HWC1
            else:
                raise ValueError(
                    "Unsupported 1-channel dtype for path input. "
                    f"Got dtype={arr.dtype} for {path}."
                )
            out = normalize_numpy_image(arr, input_format=fmt, u16_max=u16_max)
            return np.ascontiguousarray(out, dtype=np.uint8)

        if ch != 3:
            raise ValueError(f"Unsupported channel count for path input: shape={arr.shape}")

        if arr.dtype == np.uint8:
            fmt = ImageFormat.BGR_U8_HWC
        elif arr.dtype == np.uint16:
            fmt = ImageFormat.BGR_U16_HWC
        else:
            raise ValueError(
                f"Unsupported color dtype for path input. Got dtype={arr.dtype} for {path}."
            )

        out = normalize_numpy_image(arr, input_format=fmt, u16_max=u16_max)
        return np.ascontiguousarray(out, dtype=np.uint8)

    raise ValueError(
        "Unsupported image shape for path input. "
        f"Expected (H,W) or (H,W,C), got shape={arr.shape} for {path}."
    )


__all__ = ["load_path_as_rgb_u8_hwc"]
