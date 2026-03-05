from __future__ import annotations

"""Industrial defect primitives used by synthesis presets.

This module groups "defect-like" generators (rust/oil/crack/...) so they can be
reused across presets and CLI tools without duplicating logic.

The API is intentionally small and returns a `DefectResult` containing:
- `overlay_u8`: the defect texture/overlay image (same shape as input)
- `mask_u8`: binary-ish mask (H,W), uint8 {0,255}
- `meta`: JSON-friendly metadata
"""

from dataclasses import dataclass

import numpy as np

from .masks import random_crack_mask
from .perlin import fractal_perlin_noise_2d


@dataclass(frozen=True)
class DefectResult:
    overlay_u8: np.ndarray
    mask_u8: np.ndarray
    meta: dict[str, object]


def _as_u8_bgr(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        raise ValueError(f"Expected uint8 image, got dtype={arr.dtype}")
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected color image (H,W,3), got shape={arr.shape}")
    return arr


def _perlin_mask(
    h: int,
    w: int,
    *,
    rng: np.random.Generator,
    base_div: int,
    octaves: int,
    persistence: float,
    thr_range: tuple[float, float],
    blur_sigma: float,
    post_thr: float,
) -> tuple[np.ndarray, float]:
    base_res = (max(2, h // int(base_div)), max(2, w // int(base_div)))
    noise = fractal_perlin_noise_2d(
        (h, w), base_res, rng=rng, octaves=int(octaves), persistence=float(persistence)
    )
    thr = float(rng.uniform(float(thr_range[0]), float(thr_range[1])))
    mask = ((noise >= thr).astype(np.uint8) * 255).astype(np.uint8)

    try:
        import cv2  # type: ignore
    except Exception:
        cv2 = None
    if cv2 is not None and float(blur_sigma) > 0:
        blur = cv2.GaussianBlur(
            mask.astype(np.float32) / 255.0, ksize=(0, 0), sigmaX=float(blur_sigma)
        )
        mask = ((blur >= float(post_thr)).astype(np.uint8) * 255).astype(np.uint8)
    return mask, thr


def rust_defect(image_u8: np.ndarray, rng: np.random.Generator) -> DefectResult:
    """Rust/corrosion: orange-brown blobs + speckle texture."""

    img = _as_u8_bgr(image_u8)
    h, w = int(img.shape[0]), int(img.shape[1])

    mask, thr = _perlin_mask(
        h,
        w,
        rng=rng,
        base_div=96,
        octaves=3,
        persistence=0.6,
        thr_range=(0.52, 0.72),
        blur_sigma=2.0,
        post_thr=0.35,
    )

    m = mask > 0
    if not np.any(m):
        return DefectResult(
            overlay_u8=np.asarray(img, dtype=np.uint8),
            mask_u8=np.asarray(mask, dtype=np.uint8),
            meta={"defect": "rust", "skipped": True, "reason": "empty_mask"},
        )

    strength = float(rng.uniform(0.35, 1.0))
    out = img.astype(np.float32)
    rust_color = np.array([40.0, 90.0, 180.0], dtype=np.float32).reshape(1, 1, 3)  # BGR-ish
    out[m] = out[m] * (1.0 - 0.55 * strength) + rust_color * (0.55 * strength)
    speckle = rng.normal(0.0, 18.0 * strength, size=(h, w, 1)).astype(np.float32)
    out[m] = out[m] + speckle[m]
    out = np.clip(out, 0.0, 255.0).astype(np.uint8)

    return DefectResult(
        overlay_u8=np.asarray(out, dtype=np.uint8),
        mask_u8=np.asarray(mask, dtype=np.uint8),
        meta={"defect": "rust", "threshold": float(thr), "strength": float(strength)},
    )


def oil_defect(image_u8: np.ndarray, rng: np.random.Generator) -> DefectResult:
    """Oil stain: darker organic mask with subtle tint."""

    img = _as_u8_bgr(image_u8)
    h, w = int(img.shape[0]), int(img.shape[1])

    mask, thr = _perlin_mask(
        h,
        w,
        rng=rng,
        base_div=80,
        octaves=4,
        persistence=0.55,
        thr_range=(0.58, 0.80),
        blur_sigma=2.5,
        post_thr=0.45,
    )

    m = mask > 0
    if not np.any(m):
        return DefectResult(
            overlay_u8=np.asarray(img, dtype=np.uint8),
            mask_u8=np.asarray(mask, dtype=np.uint8),
            meta={"defect": "oil", "skipped": True, "reason": "empty_mask"},
        )

    strength = float(rng.uniform(0.4, 1.0))
    out = img.astype(np.float32)
    oil_color = np.array([18.0, 18.0, 28.0], dtype=np.float32).reshape(1, 1, 3)
    out[m] = out[m] * (1.0 - 0.8 * strength) + oil_color * (0.8 * strength)
    out = np.clip(out, 0.0, 255.0).astype(np.uint8)

    return DefectResult(
        overlay_u8=np.asarray(out, dtype=np.uint8),
        mask_u8=np.asarray(mask, dtype=np.uint8),
        meta={"defect": "oil", "threshold": float(thr), "strength": float(strength)},
    )


def crack_defect(image_u8: np.ndarray, rng: np.random.Generator) -> DefectResult:
    """Crack: thin, long dark fracture lines."""

    img = _as_u8_bgr(image_u8)
    h, w = int(img.shape[0]), int(img.shape[1])

    mask = random_crack_mask(
        (h, w),
        rng=rng,
        num_cracks=int(rng.integers(1, 3)),
        thickness_range=(1, 2),
        length_fraction=float(rng.uniform(0.6, 0.95)),
        curvature=float(rng.uniform(0.2, 0.5)),
        blur_sigma=float(rng.uniform(0.3, 1.2)),
        branch_prob=float(rng.uniform(0.0, 0.4)),
    )

    m = mask > 0
    strength = float(rng.uniform(0.6, 1.0))
    out = img.astype(np.float32)
    out[m] = out[m] * (1.0 - 0.75 * strength)
    out = np.clip(out, 0.0, 255.0).astype(np.uint8)

    return DefectResult(
        overlay_u8=np.asarray(out, dtype=np.uint8),
        mask_u8=np.asarray(mask, dtype=np.uint8),
        meta={"defect": "crack", "strength": float(strength)},
    )


__all__ = ["DefectResult", "rust_defect", "oil_defect", "crack_defect"]
