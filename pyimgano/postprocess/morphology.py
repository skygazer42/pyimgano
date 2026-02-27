from __future__ import annotations

"""Morphology helpers for anomaly maps and binary masks.

This module is intentionally minimal and OpenCV-backed, since OpenCV is already
used across the repo and provides stable, fast morphology primitives.
"""

from typing import Literal

import numpy as np


_KernelShape = Literal["ellipse", "rect"]
_MorphOp = Literal["open", "close", "dilate", "erode"]


def _kernel(*, ksize: int, shape: _KernelShape) -> np.ndarray:
    import cv2  # local import

    k = int(ksize)
    if k <= 0:
        raise ValueError(f"ksize must be > 0, got {ksize}")
    if shape == "ellipse":
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    if shape == "rect":
        return cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    raise ValueError(f"Unknown kernel shape: {shape!r}")


def morph_u8(
    image_u8: np.ndarray,
    *,
    op: _MorphOp,
    ksize: int,
    shape: _KernelShape = "ellipse",
) -> np.ndarray:
    """Apply morphology to an 8-bit image/mask (uint8)."""

    import cv2  # local import

    img = np.asarray(image_u8)
    if img.dtype != np.uint8:
        raise ValueError(f"Expected uint8 image, got dtype={img.dtype}")
    if img.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {img.shape}")

    kernel = _kernel(ksize=int(ksize), shape=shape)
    if op == "open":
        out = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    elif op == "close":
        out = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    elif op == "dilate":
        out = cv2.dilate(img, kernel)
    elif op == "erode":
        out = cv2.erode(img, kernel)
    else:  # pragma: no cover - guarded by Literal type
        raise ValueError(f"Unknown op: {op!r}")

    return np.asarray(out, dtype=np.uint8)


def morph_float01(
    anomaly_map: np.ndarray,
    *,
    op: Literal["open", "close"],
    ksize: int,
    shape: _KernelShape = "ellipse",
) -> np.ndarray:
    """Apply morphology to a float anomaly map in [0,1], returning float32 in [0,1]."""

    arr = np.asarray(anomaly_map, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D anomaly map, got shape {arr.shape}")

    scaled = np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8)
    out_u8 = morph_u8(scaled, op=str(op), ksize=int(ksize), shape=shape)
    return out_u8.astype(np.float32) / 255.0


def open_float01(anomaly_map: np.ndarray, *, ksize: int, shape: _KernelShape = "ellipse") -> np.ndarray:
    return morph_float01(anomaly_map, op="open", ksize=int(ksize), shape=shape)


def close_float01(anomaly_map: np.ndarray, *, ksize: int, shape: _KernelShape = "ellipse") -> np.ndarray:
    return morph_float01(anomaly_map, op="close", ksize=int(ksize), shape=shape)


__all__ = [
    "morph_u8",
    "morph_float01",
    "open_float01",
    "close_float01",
]

