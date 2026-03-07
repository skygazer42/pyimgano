from __future__ import annotations

from enum import Enum
from typing import Any, cast

import numpy as np


class ImageFormat(str, Enum):
    """Supported explicit input formats for in-memory images."""

    BGR_U8_HWC = "bgr_u8_hwc"
    GRAY_U8_HW = "gray_u8_hw"
    GRAY_U8_HWC1 = "gray_u8_hwc1"
    GRAY_U16_HW = "gray_u16_hw"
    GRAY_U16_HWC1 = "gray_u16_hwc1"
    RGB_U8_HWC = "rgb_u8_hwc"
    BGR_U16_HWC = "bgr_u16_hwc"
    RGB_U16_HWC = "rgb_u16_hwc"
    BGR_F32_HWC = "bgr_f32_hwc"
    RGB_F32_HWC = "rgb_f32_hwc"
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


def _resolve_u16_max(u16_max: int | None) -> int:
    if u16_max is None:
        return 65535
    m = int(u16_max)
    if m <= 0 or m > 65535:
        raise ValueError(f"u16_max must be in [1, 65535], got {u16_max!r}")
    return m


def _scale_u16_to_u8(arr_u16: np.ndarray, *, u16_max: int) -> np.ndarray:
    # Use float64 so scaling stays stable for large values.
    scaled = np.clip(
        np.rint(arr_u16.astype(np.float64) * (255.0 / float(u16_max))), 0.0, 255.0
    ).astype(np.uint8, copy=False)
    return cast(np.ndarray, np.ascontiguousarray(scaled))


def normalize_numpy_image(
    image: Any, *, input_format: str | ImageFormat, u16_max: int | None = None
) -> np.ndarray:
    """Normalize an in-memory image into canonical ``RGB/u8/HWC``.

    This function is intentionally strict: it uses the declared `input_format`
    and does not guess.
    """

    fmt = parse_image_format(input_format)
    arr = _require_ndarray(image)

    if fmt is ImageFormat.GRAY_U8_HW:
        if arr.dtype != np.uint8:
            raise ValueError(f"Expected dtype=uint8 for {fmt.value}, got {arr.dtype}")
        if arr.ndim != 2:
            raise ValueError(f"Expected shape (H,W) for {fmt.value}, got {arr.shape}")
        rgb = np.repeat(arr[:, :, None], 3, axis=2)
        return cast(np.ndarray, np.ascontiguousarray(rgb))

    if fmt is ImageFormat.GRAY_U8_HWC1:
        if arr.dtype != np.uint8:
            raise ValueError(f"Expected dtype=uint8 for {fmt.value}, got {arr.dtype}")
        if arr.ndim != 3 or arr.shape[2] != 1:
            raise ValueError(f"Expected shape (H,W,1) for {fmt.value}, got {arr.shape}")
        rgb = np.repeat(arr, 3, axis=2)
        return cast(np.ndarray, np.ascontiguousarray(rgb))

    if fmt is ImageFormat.GRAY_U16_HW:
        if arr.dtype != np.uint16:
            raise ValueError(f"Expected dtype=uint16 for {fmt.value}, got {arr.dtype}")
        if arr.ndim != 2:
            raise ValueError(f"Expected shape (H,W) for {fmt.value}, got {arr.shape}")
        u16_max_val = _resolve_u16_max(u16_max)
        u8 = _scale_u16_to_u8(arr, u16_max=u16_max_val)
        rgb = np.repeat(u8[:, :, None], 3, axis=2)
        return cast(np.ndarray, np.ascontiguousarray(rgb))

    if fmt is ImageFormat.GRAY_U16_HWC1:
        if arr.dtype != np.uint16:
            raise ValueError(f"Expected dtype=uint16 for {fmt.value}, got {arr.dtype}")
        if arr.ndim != 3 or arr.shape[2] != 1:
            raise ValueError(f"Expected shape (H,W,1) for {fmt.value}, got {arr.shape}")
        u16_max_val = _resolve_u16_max(u16_max)
        u8 = _scale_u16_to_u8(arr, u16_max=u16_max_val)
        rgb = np.repeat(u8, 3, axis=2)
        return cast(np.ndarray, np.ascontiguousarray(rgb))

    if fmt is ImageFormat.BGR_U8_HWC:
        if arr.dtype != np.uint8:
            raise ValueError(f"Expected dtype=uint8 for {fmt.value}, got {arr.dtype}")
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Expected shape (H,W,3) for {fmt.value}, got {arr.shape}")
        rgb = arr[..., ::-1]
        return cast(np.ndarray, np.ascontiguousarray(rgb))

    if fmt is ImageFormat.RGB_U8_HWC:
        if arr.dtype != np.uint8:
            raise ValueError(f"Expected dtype=uint8 for {fmt.value}, got {arr.dtype}")
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Expected shape (H,W,3) for {fmt.value}, got {arr.shape}")
        return cast(np.ndarray, np.ascontiguousarray(arr))

    if fmt in (ImageFormat.BGR_U16_HWC, ImageFormat.RGB_U16_HWC):
        if arr.dtype != np.uint16:
            raise ValueError(f"Expected dtype=uint16 for {fmt.value}, got {arr.dtype}")
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Expected shape (H,W,3) for {fmt.value}, got {arr.shape}")
        u16_max_val = _resolve_u16_max(u16_max)
        hwc = arr
        if fmt is ImageFormat.BGR_U16_HWC:
            hwc = hwc[..., ::-1]
        u8 = _scale_u16_to_u8(hwc, u16_max=u16_max_val)
        return cast(np.ndarray, np.ascontiguousarray(u8))

    if fmt in (ImageFormat.BGR_F32_HWC, ImageFormat.RGB_F32_HWC):
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Expected shape (H,W,3) for {fmt.value}, got {arr.shape}")
        if arr.dtype not in (np.float32, np.float64):
            raise ValueError(f"Expected dtype=float32/float64 for {fmt.value}, got {arr.dtype}")
        if arr.size == 0:
            raise ValueError(f"Empty image is not supported for {fmt.value}")
        max_val = float(np.max(arr))
        min_val = float(np.min(arr))
        if max_val > 1.0 + 1e-6 or min_val < 0.0 - 1e-6:
            raise ValueError(
                f"Expected values in [0,1] for {fmt.value}. Got min={min_val:.6f}, max={max_val:.6f}."
            )
        hwc = arr
        if fmt is ImageFormat.BGR_F32_HWC:
            hwc = hwc[..., ::-1]
        scaled = np.clip(np.rint(hwc * 255.0), 0.0, 255.0).astype(np.uint8, copy=False)
        return cast(np.ndarray, np.ascontiguousarray(scaled))

    if fmt is ImageFormat.RGB_F32_CHW:
        if arr.ndim != 3 or arr.shape[0] != 3:
            raise ValueError(f"Expected shape (3,H,W) for {fmt.value}, got {arr.shape}")
        if arr.dtype not in (np.float32, np.float64):
            raise ValueError(f"Expected dtype=float32/float64 for {fmt.value}, got {arr.dtype}")
        if arr.size == 0:
            raise ValueError(f"Empty image is not supported for {fmt.value}")
        max_val = float(np.max(arr))
        min_val = float(np.min(arr))
        if max_val > 1.0 + 1e-6 or min_val < 0.0 - 1e-6:
            raise ValueError(
                f"Expected values in [0,1] for {fmt.value}. Got min={min_val:.6f}, max={max_val:.6f}."
            )
        hwc = np.transpose(arr, (1, 2, 0))
        scaled = np.clip(np.rint(hwc * 255.0), 0.0, 255.0).astype(np.uint8, copy=False)
        return cast(np.ndarray, np.ascontiguousarray(scaled))

    raise RuntimeError(f"Unhandled image format: {fmt}")
