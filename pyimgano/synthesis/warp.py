from __future__ import annotations

from typing import Optional

import numpy as np


def _as_u8_image(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        raise ValueError(f"Expected uint8 image, got dtype={arr.dtype}")
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[2] == 3:
        return arr
    raise ValueError(f"Expected grayscale (H,W) or color (H,W,3) image, got {arr.shape}")


def apply_slight_warp(
    image_u8: np.ndarray,
    *,
    rng: np.random.Generator,
    max_rotate_deg: float = 2.0,
    max_translate_frac: float = 0.02,
    max_scale_delta: float = 0.02,
    interpolation: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Apply a slight geometric misalignment (affine warp).

    Notes
    -----
    - Intended as a **stress test** for template/reference style inspection.
    - Uses reflect border to avoid obvious black padding artifacts.
    - Returns a dense mask because most pixels are displaced.
    """

    import cv2  # local import (opencv is a core dependency)

    img = _as_u8_image(image_u8)
    h, w = int(img.shape[0]), int(img.shape[1])
    if h <= 0 or w <= 0:
        empty = np.zeros((max(0, h), max(0, w)), dtype=np.uint8)
        return np.asarray(img, dtype=np.uint8), empty, {"skipped": True, "reason": "empty_image"}

    angle = float(rng.uniform(-abs(float(max_rotate_deg)), abs(float(max_rotate_deg))))
    scale = 1.0 + float(rng.uniform(-abs(float(max_scale_delta)), abs(float(max_scale_delta))))
    tx = float(
        rng.uniform(-abs(float(max_translate_frac)), abs(float(max_translate_frac)))
    ) * float(w)
    ty = float(
        rng.uniform(-abs(float(max_translate_frac)), abs(float(max_translate_frac)))
    ) * float(h)

    center = (float(w - 1) * 0.5, float(h - 1) * 0.5)
    m = cv2.getRotationMatrix2D(center, angle, scale).astype(np.float32)
    m[0, 2] += tx
    m[1, 2] += ty

    interp = cv2.INTER_LINEAR if interpolation is None else int(interpolation)
    warped = cv2.warpAffine(
        img,
        m,
        dsize=(w, h),
        flags=interp,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    overlay = np.asarray(warped, dtype=np.uint8)
    mask = np.full((h, w), 255, dtype=np.uint8)
    meta: dict[str, object] = {
        "mode": "affine",
        "angle_deg": float(angle),
        "scale": float(scale),
        "translate_xy": (float(tx), float(ty)),
    }
    return overlay, mask, meta


__all__ = ["apply_slight_warp"]
