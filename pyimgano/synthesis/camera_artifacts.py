from __future__ import annotations

from typing import Optional

import numpy as np


def _as_u8_image(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        raise ValueError(f"Expected uint8 image, got dtype={arr.dtype}")
    if arr.ndim not in (2, 3):
        raise ValueError(f"Expected grayscale (H,W) or color (H,W,3) image, got {arr.shape}")
    if arr.ndim == 3 and arr.shape[2] != 3:
        raise ValueError(f"Expected grayscale (H,W) or color (H,W,3) image, got {arr.shape}")
    return arr


def apply_defocus_blur(
    image_u8: np.ndarray,
    *,
    rng: np.random.Generator,
    strength: Optional[float] = None,
    sigma_range: tuple[float, float] = (0.6, 3.6),
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Apply defocus-like blur (Gaussian) and return (overlay, mask, meta).

    Notes
    -----
    - Uses OpenCV when available.
    - Returns a dense mask because blur affects most pixels.
    """

    import cv2  # local import (opencv is a core dependency)

    img = _as_u8_image(image_u8)
    h, w = int(img.shape[0]), int(img.shape[1])
    if h <= 0 or w <= 0:
        empty = np.zeros((max(0, h), max(0, w)), dtype=np.uint8)
        return np.asarray(img, dtype=np.uint8), empty, {"skipped": True, "reason": "empty_image"}

    s = float(rng.uniform(0.25, 0.9) if strength is None else strength)
    s = float(np.clip(s, 0.0, 1.0))
    lo, hi = float(sigma_range[0]), float(sigma_range[1])
    if not (0.0 < lo <= hi):
        raise ValueError(f"sigma_range must satisfy 0 < lo <= hi, got {sigma_range}")

    sigma = float(rng.uniform(lo, hi)) * float(s)
    sigma = float(np.clip(sigma, 0.0, hi))

    if sigma <= 1e-6:
        overlay = np.asarray(img, dtype=np.uint8)
    else:
        overlay = cv2.GaussianBlur(img, ksize=(0, 0), sigmaX=float(sigma), sigmaY=float(sigma))
        overlay = np.asarray(overlay, dtype=np.uint8)

    mask = np.full((h, w), 255, dtype=np.uint8)
    meta: dict[str, object] = {
        "mode": "defocus",
        "strength": float(s),
        "sigma": float(sigma),
    }
    return overlay, mask, meta


def apply_lens_distortion(
    image_u8: np.ndarray,
    *,
    rng: np.random.Generator,
    strength: Optional[float] = None,
    k1_range: tuple[float, float] = (-0.25, 0.25),
    center_jitter: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Apply a simple radial lens distortion via coordinate remapping.

    Parameters
    ----------
    k1_range:
        Range for the radial distortion coefficient `k1`. Positive/negative
        values produce different warping directions (barrel vs pincushion-like).
    center_jitter:
        Amount of random center jitter as a fraction of width/height.

    Notes
    -----
    - Returns a dense mask because most pixels are displaced.
    - Uses reflect borders to avoid black padding artifacts.
    """

    import cv2  # local import (opencv is a core dependency)

    img = _as_u8_image(image_u8)
    h, w = int(img.shape[0]), int(img.shape[1])
    if h <= 0 or w <= 0:
        empty = np.zeros((max(0, h), max(0, w)), dtype=np.uint8)
        return np.asarray(img, dtype=np.uint8), empty, {"skipped": True, "reason": "empty_image"}

    s = float(rng.uniform(0.25, 0.9) if strength is None else strength)
    s = float(np.clip(s, 0.0, 1.0))

    k1_lo, k1_hi = float(k1_range[0]), float(k1_range[1])
    if not (k1_lo <= k1_hi):
        raise ValueError(f"k1_range must satisfy lo <= hi, got {k1_range}")
    k1 = float(rng.uniform(k1_lo, k1_hi)) * float(s)

    # Center jitter (simulate imperfect optical axis / alignment).
    jitter = float(np.clip(center_jitter, 0.0, 0.25))
    cx = (float(w - 1) * 0.5) + float(rng.uniform(-jitter, jitter)) * float(w)
    cy = (float(h - 1) * 0.5) + float(rng.uniform(-jitter, jitter)) * float(h)

    # Build inverse mapping (output -> source) using a simple radial model.
    yy, xx = np.meshgrid(
        np.arange(h, dtype=np.float32),
        np.arange(w, dtype=np.float32),
        indexing="ij",
    )
    denom_x = max(1.0, float(w - 1) * 0.5)
    denom_y = max(1.0, float(h - 1) * 0.5)
    x = (xx - float(cx)) / float(denom_x)
    y = (yy - float(cy)) / float(denom_y)
    r2 = x * x + y * y
    factor = (1.0 + float(k1) * r2).astype(np.float32, copy=False)

    map_x = (x * factor) * float(denom_x) + float(cx)
    map_y = (y * factor) * float(denom_y) + float(cy)

    distorted = cv2.remap(
        img,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    overlay = np.asarray(distorted, dtype=np.uint8)
    mask = np.full((h, w), 255, dtype=np.uint8)
    meta: dict[str, object] = {
        "mode": "lens_distortion",
        "strength": float(s),
        "k1": float(k1),
        "center_xy": (float(cx), float(cy)),
    }
    return overlay, mask, meta


__all__ = ["apply_defocus_blur", "apply_lens_distortion"]
