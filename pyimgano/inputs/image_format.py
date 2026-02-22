from __future__ import annotations

from enum import Enum
from typing import Any

import numpy as np


class ImageFormat(str, Enum):
    """Supported explicit input formats for in-memory images."""

    BGR_U8_HWC = "bgr_u8_hwc"
    RGB_U8_HWC = "rgb_u8_hwc"
    RGB_F32_CHW = "rgb_f32_chw"


def parse_image_format(raw: str | ImageFormat) -> ImageFormat:
    if isinstance(raw, ImageFormat):
        return raw
    try:
        return ImageFormat(str(raw))
    except Exception as exc:  # noqa: BLE001 - value validation helper
        raise ValueError(f"Unknown image format: {raw!r}") from exc


def _require_ndarray(image: Any) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(image)}")
    return image


def normalize_numpy_image(image: Any, *, input_format: str | ImageFormat) -> np.ndarray:
    """Normalize an in-memory image into canonical ``RGB/u8/HWC``.

    This function is intentionally strict: it uses the declared `input_format`
    and does not guess.
    """

    fmt = parse_image_format(input_format)
    arr = _require_ndarray(image)

    if fmt is ImageFormat.BGR_U8_HWC:
        if arr.dtype != np.uint8:
            raise ValueError(f"Expected dtype=uint8 for {fmt.value}, got {arr.dtype}")
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Expected shape (H,W,3) for {fmt.value}, got {arr.shape}")
        rgb = arr[..., ::-1]
        return np.ascontiguousarray(rgb)

    if fmt is ImageFormat.RGB_U8_HWC:
        if arr.dtype != np.uint8:
            raise ValueError(f"Expected dtype=uint8 for {fmt.value}, got {arr.dtype}")
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Expected shape (H,W,3) for {fmt.value}, got {arr.shape}")
        return np.ascontiguousarray(arr)

    if fmt is ImageFormat.RGB_F32_CHW:
        if arr.ndim != 3 or arr.shape[0] != 3:
            raise ValueError(f"Expected shape (3,H,W) for {fmt.value}, got {arr.shape}")
        if arr.dtype not in (np.float32, np.float64):
            raise ValueError(f"Expected dtype=float32/float64 for {fmt.value}, got {arr.dtype}")
        max_val = float(np.max(arr))
        min_val = float(np.min(arr))
        if max_val > 1.0 + 1e-6 or min_val < 0.0 - 1e-6:
            raise ValueError(
                f"Expected values in [0,1] for {fmt.value}. Got min={min_val:.6f}, max={max_val:.6f}."
            )
        hwc = np.transpose(arr, (1, 2, 0))
        scaled = np.clip(np.rint(hwc * 255.0), 0.0, 255.0).astype(np.uint8, copy=False)
        return np.ascontiguousarray(scaled)

    raise RuntimeError(f"Unhandled image format: {fmt}")
