from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from pyimgano.utils.image_u8 import as_u8_image

from .masks import ensure_u8_mask

_Variant = Literal["normal", "scar", "3way"]


@dataclass(frozen=True)
class CutPasteConfig:
    area_ratio_range: tuple[float, float] = (0.02, 0.15)
    aspect_ratio_range: tuple[float, float] = (0.3, 3.3)
    min_size: int = 8
    color_jitter_strength: float = 0.25


def _sample_patch_hw(
    h: int,
    w: int,
    *,
    rng: np.random.Generator,
    area_ratio_range: tuple[float, float],
    aspect_ratio_range: tuple[float, float],
    min_size: int,
) -> tuple[int, int]:
    lo, hi = float(area_ratio_range[0]), float(area_ratio_range[1])
    lo = max(1e-6, lo)
    hi = max(lo, hi)
    r = float(rng.uniform(lo, hi))
    area = r * float(h * w)

    ar_lo, ar_hi = float(aspect_ratio_range[0]), float(aspect_ratio_range[1])
    ar_lo = max(1e-3, ar_lo)
    ar_hi = max(ar_lo, ar_hi)
    ar = float(rng.uniform(ar_lo, ar_hi))

    pw = int(round(np.sqrt(area * ar)))
    ph = int(round(np.sqrt(area / ar)))

    pw = int(np.clip(pw, int(min_size), max(int(min_size), w)))
    ph = int(np.clip(ph, int(min_size), max(int(min_size), h)))
    return ph, pw


def _jitter_patch(patch: np.ndarray, *, rng: np.random.Generator, strength: float) -> np.ndarray:
    if float(strength) <= 0.0:
        return patch

    delta = float(rng.uniform(-strength, strength))
    shift = float(rng.uniform(-strength, strength)) * 255.0 * 0.25

    out = patch.astype(np.float32)
    out = out * (1.0 + delta) + shift
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def cutpaste(
    image_u8: np.ndarray,
    *,
    rng: np.random.Generator,
    variant: _Variant = "normal",
    config: CutPasteConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a CutPaste-style synthetic anomaly to an image.

    Returns
    -------
        out_image_u8, mask_u8
        mask is uint8 (H,W) with {0,255}.
    """

    img = as_u8_image(image_u8)
    cfg = CutPasteConfig() if config is None else config
    h, w = int(img.shape[0]), int(img.shape[1])
    if h < 2 or w < 2:
        return np.asarray(img, dtype=np.uint8), np.zeros((h, w), dtype=np.uint8)

    v = str(variant)
    if v == "3way":
        v = "scar" if bool(rng.integers(0, 2)) else "normal"

    if v == "scar":
        # A thin, long patch.
        ph = max(int(cfg.min_size), int(round(0.06 * h)))
        pw = max(int(cfg.min_size), int(round(0.35 * w)))
        if bool(rng.integers(0, 2)):
            ph, pw = pw, ph  # swap orientation
        ph = int(np.clip(ph, int(cfg.min_size), h))
        pw = int(np.clip(pw, int(cfg.min_size), w))
    elif v == "normal":
        ph, pw = _sample_patch_hw(
            h,
            w,
            rng=rng,
            area_ratio_range=cfg.area_ratio_range,
            aspect_ratio_range=cfg.aspect_ratio_range,
            min_size=int(cfg.min_size),
        )
    else:
        raise ValueError(f"Unknown CutPaste variant: {variant!r}")

    y0 = int(rng.integers(0, max(1, h - ph + 1)))
    x0 = int(rng.integers(0, max(1, w - pw + 1)))
    y1 = int(rng.integers(0, max(1, h - ph + 1)))
    x1 = int(rng.integers(0, max(1, w - pw + 1)))

    patch = np.asarray(img[y0 : y0 + ph, x0 : x0 + pw], dtype=np.uint8)
    patch = _jitter_patch(patch, rng=rng, strength=float(cfg.color_jitter_strength))

    # Optional rotation.
    rot_k = int(rng.integers(0, 4))
    if rot_k:
        patch = np.rot90(patch, k=rot_k).copy()
        ph, pw = int(patch.shape[0]), int(patch.shape[1])
        # Re-sample destination if shape changed.
        y1 = int(rng.integers(0, max(1, h - ph + 1)))
        x1 = int(rng.integers(0, max(1, w - pw + 1)))

    out = np.array(img, copy=True)
    out[y1 : y1 + ph, x1 : x1 + pw] = patch

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y1 : y1 + ph, x1 : x1 + pw] = 255

    return np.asarray(out, dtype=np.uint8), ensure_u8_mask(mask)
