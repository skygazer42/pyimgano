from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .masks import (
    ScratchSpec,
    random_blob_mask,
    random_ellipse_mask,
    random_crack_mask,
    random_curve_scratch_mask,
    random_scratch_mask,
)
from .perlin import fractal_perlin_noise_2d


@dataclass(frozen=True)
class PresetResult:
    overlay_u8: np.ndarray
    mask_u8: np.ndarray
    meta: dict[str, object]


PresetFn = Callable[[np.ndarray, np.random.Generator], PresetResult]


def _as_u8_image(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        raise ValueError(f"Expected uint8 image, got dtype={arr.dtype}")
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[2] == 3:
        return arr
    raise ValueError(f"Expected grayscale (H,W) or color (H,W,3) image, got {arr.shape}")


def _apply_tint(
    base: np.ndarray,
    mask_u8: np.ndarray,
    *,
    rng: np.random.Generator,
    strength: float,
    mode: str,
) -> np.ndarray:
    img = np.asarray(base, dtype=np.uint8)
    m = (np.asarray(mask_u8) > 0)
    if not np.any(m):
        return img

    s = float(np.clip(strength, 0.0, 1.0))
    out = img.astype(np.float32)

    if out.ndim == 2:
        if mode == "darken":
            out[m] = out[m] * (1.0 - 0.7 * s)
        elif mode == "brighten":
            out[m] = out[m] + (255.0 - out[m]) * (0.8 * s)
        else:
            raise ValueError(f"Unknown mode: {mode!r}")
        return np.clip(out, 0.0, 255.0).astype(np.uint8)

    # Color tint (BGR-ish, but visually fine either way for synthesis).
    tint = rng.uniform(0.0, 255.0, size=(3,)).astype(np.float32)
    tint = (tint * 0.15 + np.array([30.0, 60.0, 90.0], dtype=np.float32) * 0.85).astype(
        np.float32
    )

    if mode == "stain":
        out[m] = out[m] * (1.0 - 0.35 * s) + tint * (0.35 * s)
    elif mode == "darken":
        out[m] = out[m] * (1.0 - 0.75 * s)
    elif mode == "brighten":
        out[m] = out[m] + (255.0 - out[m]) * (0.85 * s)
    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def _preset_scratch(image_u8: np.ndarray, rng: np.random.Generator) -> PresetResult:
    img = _as_u8_image(image_u8)
    h, w = int(img.shape[0]), int(img.shape[1])
    spec = ScratchSpec(
        thickness=int(rng.integers(1, 4)),
        length_fraction=float(rng.uniform(0.25, 0.65)),
        jitter=float(rng.uniform(0.05, 0.25)),
        blur_sigma=float(rng.uniform(0.0, 1.2)),
    )
    # v2 scratch: sometimes use curved random-walk scratches (more "industrial").
    if float(rng.uniform(0.0, 1.0)) < 0.5:
        mask = random_curve_scratch_mask(
            (h, w), rng=rng, num_scratches=int(rng.integers(1, 4)), spec=spec, curvature=0.25
        )
        variant = "curve"
    else:
        mask = random_scratch_mask((h, w), rng=rng, num_scratches=int(rng.integers(1, 4)), spec=spec)
        variant = "line"
    strength = float(rng.uniform(0.4, 1.0))
    overlay = _apply_tint(img, mask, rng=rng, strength=strength, mode="darken")
    return PresetResult(
        overlay_u8=np.asarray(overlay, dtype=np.uint8),
        mask_u8=np.asarray(mask, dtype=np.uint8),
        meta={"preset": "scratch", "strength": strength, "variant": variant},
    )


def _preset_stain(image_u8: np.ndarray, rng: np.random.Generator) -> PresetResult:
    img = _as_u8_image(image_u8)
    h, w = int(img.shape[0]), int(img.shape[1])

    # Perlin-based mask tends to create organic stains.
    base_res = (max(2, h // 64), max(2, w // 64))
    noise = fractal_perlin_noise_2d((h, w), base_res, rng=rng, octaves=4, persistence=0.55)
    thr = float(rng.uniform(0.55, 0.78))
    mask = ((noise >= thr).astype(np.uint8) * 255).astype(np.uint8)

    # Smooth and tighten the mask a bit.
    try:
        import cv2  # type: ignore
    except Exception:
        cv2 = None
    if cv2 is not None:
        blur = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, ksize=(0, 0), sigmaX=2.0)
        mask = ((blur >= 0.5).astype(np.uint8) * 255).astype(np.uint8)

    strength = float(rng.uniform(0.35, 1.0))
    overlay = _apply_tint(img, mask, rng=rng, strength=strength, mode="stain")
    return PresetResult(
        overlay_u8=np.asarray(overlay, dtype=np.uint8),
        mask_u8=np.asarray(mask, dtype=np.uint8),
        meta={"preset": "stain", "threshold": thr, "strength": strength},
    )


def _preset_pit(image_u8: np.ndarray, rng: np.random.Generator) -> PresetResult:
    img = _as_u8_image(image_u8)
    h, w = int(img.shape[0]), int(img.shape[1])

    radius = max(2, int(round(min(h, w) * float(rng.uniform(0.01, 0.05)))))
    mask = random_blob_mask(
        (h, w),
        rng=rng,
        num_blobs=int(rng.integers(1, 4)),
        radius_range=(max(2, radius // 2), max(3, radius)),
        blur_sigma=float(rng.uniform(0.8, 2.0)),
        threshold=0.4,
    )
    strength = float(rng.uniform(0.5, 1.0))
    overlay = _apply_tint(img, mask, rng=rng, strength=strength, mode="darken")
    return PresetResult(
        overlay_u8=np.asarray(overlay, dtype=np.uint8),
        mask_u8=np.asarray(mask, dtype=np.uint8),
        meta={"preset": "pit", "strength": strength},
    )


def _preset_glare(image_u8: np.ndarray, rng: np.random.Generator) -> PresetResult:
    img = _as_u8_image(image_u8)
    h, w = int(img.shape[0]), int(img.shape[1])

    mask = random_ellipse_mask(
        (h, w),
        rng=rng,
        num_ellipses=int(rng.integers(1, 3)),
        axis_range=(max(6, min(h, w) // 12), max(8, min(h, w) // 5)),
        blur_sigma=float(rng.uniform(1.0, 3.0)),
        threshold=0.4,
    )
    strength = float(rng.uniform(0.3, 1.0))
    overlay = _apply_tint(img, mask, rng=rng, strength=strength, mode="brighten")
    return PresetResult(
        overlay_u8=np.asarray(overlay, dtype=np.uint8),
        mask_u8=np.asarray(mask, dtype=np.uint8),
        meta={"preset": "glare", "strength": strength},
    )


def _preset_rust(image_u8: np.ndarray, rng: np.random.Generator) -> PresetResult:
    """Rust/corrosion: orange-brown organic blobs + speckle texture."""

    img = _as_u8_image(image_u8)
    h, w = int(img.shape[0]), int(img.shape[1])

    base_res = (max(2, h // 96), max(2, w // 96))
    noise = fractal_perlin_noise_2d((h, w), base_res, rng=rng, octaves=3, persistence=0.6)
    thr = float(rng.uniform(0.52, 0.72))
    mask = ((noise >= thr).astype(np.uint8) * 255).astype(np.uint8)

    try:
        import cv2  # type: ignore
    except Exception:
        cv2 = None
    if cv2 is not None:
        mask = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, ksize=(0, 0), sigmaX=2.0)
        mask = ((mask >= 0.35).astype(np.uint8) * 255).astype(np.uint8)

    m = (mask > 0)
    if not np.any(m):
        return PresetResult(
            overlay_u8=np.asarray(img, dtype=np.uint8),
            mask_u8=np.asarray(mask, dtype=np.uint8),
            meta={"preset": "rust", "skipped": True, "reason": "empty_mask"},
        )

    strength = float(rng.uniform(0.35, 1.0))
    out = img.astype(np.float32)
    rust_color = np.array([40.0, 90.0, 180.0], dtype=np.float32).reshape(1, 1, 3)  # BGR-ish
    out[m] = out[m] * (1.0 - 0.55 * strength) + rust_color * (0.55 * strength)
    # Add speckle texture inside mask.
    speckle = rng.normal(0.0, 18.0 * strength, size=(h, w, 1)).astype(np.float32)
    out[m] = out[m] + speckle[m]
    out = np.clip(out, 0.0, 255.0).astype(np.uint8)

    return PresetResult(
        overlay_u8=np.asarray(out, dtype=np.uint8),
        mask_u8=np.asarray(mask, dtype=np.uint8),
        meta={"preset": "rust", "threshold": thr, "strength": strength},
    )


def _preset_oil(image_u8: np.ndarray, rng: np.random.Generator) -> PresetResult:
    """Oil stain: darker organic mask with subtle tint."""

    img = _as_u8_image(image_u8)
    h, w = int(img.shape[0]), int(img.shape[1])

    base_res = (max(2, h // 80), max(2, w // 80))
    noise = fractal_perlin_noise_2d((h, w), base_res, rng=rng, octaves=4, persistence=0.55)
    thr = float(rng.uniform(0.58, 0.8))
    mask = ((noise >= thr).astype(np.uint8) * 255).astype(np.uint8)

    try:
        import cv2  # type: ignore
    except Exception:
        cv2 = None
    if cv2 is not None:
        mask = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, ksize=(0, 0), sigmaX=2.5)
        mask = ((mask >= 0.45).astype(np.uint8) * 255).astype(np.uint8)

    m = (mask > 0)
    if not np.any(m):
        return PresetResult(
            overlay_u8=np.asarray(img, dtype=np.uint8),
            mask_u8=np.asarray(mask, dtype=np.uint8),
            meta={"preset": "oil", "skipped": True, "reason": "empty_mask"},
        )

    strength = float(rng.uniform(0.4, 1.0))
    out = img.astype(np.float32)
    oil_color = np.array([18.0, 18.0, 28.0], dtype=np.float32).reshape(1, 1, 3)
    out[m] = out[m] * (1.0 - 0.8 * strength) + oil_color * (0.8 * strength)
    out = np.clip(out, 0.0, 255.0).astype(np.uint8)

    return PresetResult(
        overlay_u8=np.asarray(out, dtype=np.uint8),
        mask_u8=np.asarray(mask, dtype=np.uint8),
        meta={"preset": "oil", "threshold": thr, "strength": strength},
    )


def _preset_crack(image_u8: np.ndarray, rng: np.random.Generator) -> PresetResult:
    """Crack: thin, long dark fracture lines."""

    img = _as_u8_image(image_u8)
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
    strength = float(rng.uniform(0.6, 1.0))
    overlay = _apply_tint(img, mask, rng=rng, strength=strength, mode="darken")
    return PresetResult(
        overlay_u8=np.asarray(overlay, dtype=np.uint8),
        mask_u8=np.asarray(mask, dtype=np.uint8),
        meta={"preset": "crack", "strength": strength},
    )


_PRESETS: dict[str, PresetFn] = {
    "scratch": _preset_scratch,
    "stain": _preset_stain,
    "pit": _preset_pit,
    "glare": _preset_glare,
    "rust": _preset_rust,
    "oil": _preset_oil,
    "crack": _preset_crack,
}


def get_preset_names() -> list[str]:
    return sorted(_PRESETS.keys())


def make_preset(name: str) -> PresetFn:
    key = str(name).strip().lower()
    fn = _PRESETS.get(key)
    if fn is None:
        raise ValueError(
            f"Unknown synthesis preset: {name!r}. Available: {', '.join(get_preset_names())}"
        )
    return fn
