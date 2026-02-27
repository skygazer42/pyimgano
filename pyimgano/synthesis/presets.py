from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .masks import (
    ScratchSpec,
    random_blob_mask,
    random_brush_stroke_mask,
    random_edge_band_mask,
    random_ellipse_mask,
    random_crack_mask,
    random_curve_scratch_mask,
    random_scratch_mask,
    random_spatter_mask,
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


def _preset_brush(image_u8: np.ndarray, rng: np.random.Generator) -> PresetResult:
    """Brush/paint strokes: thick organic streaks."""

    img = _as_u8_image(image_u8)
    h, w = int(img.shape[0]), int(img.shape[1])

    mask = random_brush_stroke_mask(
        (h, w),
        rng=rng,
        num_strokes=int(rng.integers(1, 4)),
        thickness_range=(max(4, min(h, w) // 18), max(8, min(h, w) // 7)),
        blur_sigma=float(rng.uniform(0.8, 2.2)),
        threshold=0.2,
    )
    strength = float(rng.uniform(0.45, 1.0))
    # Brush strokes usually have some tint rather than pure darkening.
    overlay = _apply_tint(img, mask, rng=rng, strength=strength, mode="stain")
    return PresetResult(
        overlay_u8=np.asarray(overlay, dtype=np.uint8),
        mask_u8=np.asarray(mask, dtype=np.uint8),
        meta={"preset": "brush", "strength": strength},
    )


def _preset_spatter(image_u8: np.ndarray, rng: np.random.Generator) -> PresetResult:
    """Droplets / spatter: many tiny blobs."""

    img = _as_u8_image(image_u8)
    h, w = int(img.shape[0]), int(img.shape[1])

    mask = random_spatter_mask(
        (h, w),
        rng=rng,
        num_droplets=int(rng.integers(24, 96)),
        radius_range=(1, max(2, int(round(min(h, w) * 0.02)))),
        blur_sigma=float(rng.uniform(0.6, 1.6)),
        threshold=0.25,
    )
    strength = float(rng.uniform(0.35, 1.0))
    overlay = _apply_tint(img, mask, rng=rng, strength=strength, mode="stain")
    return PresetResult(
        overlay_u8=np.asarray(overlay, dtype=np.uint8),
        mask_u8=np.asarray(mask, dtype=np.uint8),
        meta={"preset": "spatter", "strength": strength},
    )


def _preset_tape(image_u8: np.ndarray, rng: np.random.Generator) -> PresetResult:
    """Tape/patch defect: rectangular-ish region with slight blur."""

    import cv2  # local import

    img = _as_u8_image(image_u8)
    h, w = int(img.shape[0]), int(img.shape[1])

    mask = np.zeros((h, w), dtype=np.uint8)
    n = int(rng.integers(1, 3))
    for _ in range(n):
        ph = int(round(float(rng.uniform(0.12, 0.45)) * float(h)))
        pw = int(round(float(rng.uniform(0.12, 0.45)) * float(w)))
        ph = int(np.clip(ph, 4, h))
        pw = int(np.clip(pw, 4, w))
        y0 = int(rng.integers(0, max(1, h - ph + 1)))
        x0 = int(rng.integers(0, max(1, w - pw + 1)))
        cv2.rectangle(mask, (x0, y0), (x0 + pw - 1, y0 + ph - 1), color=255, thickness=-1)

    sigma = float(rng.uniform(0.8, 2.5))
    blur = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, ksize=(0, 0), sigmaX=sigma)
    mask = ((blur >= 0.35).astype(np.uint8) * 255).astype(np.uint8)

    strength = float(rng.uniform(0.35, 1.0))
    # Tape can be both brighter (gloss) or darker (dirt). Randomize.
    mode = "brighten" if float(rng.uniform(0.0, 1.0)) < 0.4 else "stain"
    overlay = _apply_tint(img, mask, rng=rng, strength=strength, mode=mode)
    return PresetResult(
        overlay_u8=np.asarray(overlay, dtype=np.uint8),
        mask_u8=np.asarray(mask, dtype=np.uint8),
        meta={"preset": "tape", "strength": strength, "mode": mode},
    )


def _preset_marker(image_u8: np.ndarray, rng: np.random.Generator) -> PresetResult:
    """Marker stroke: usually a single large brush stroke, stronger tint."""

    img = _as_u8_image(image_u8)
    h, w = int(img.shape[0]), int(img.shape[1])

    mask = random_brush_stroke_mask(
        (h, w),
        rng=rng,
        num_strokes=1,
        thickness_range=(max(8, min(h, w) // 10), max(12, min(h, w) // 5)),
        curvature=float(rng.uniform(0.2, 0.6)),
        jitter=float(rng.uniform(0.2, 0.6)),
        blur_sigma=float(rng.uniform(0.6, 1.8)),
        threshold=0.2,
    )
    strength = float(rng.uniform(0.55, 1.0))
    overlay = _apply_tint(img, mask, rng=rng, strength=strength, mode="stain")
    return PresetResult(
        overlay_u8=np.asarray(overlay, dtype=np.uint8),
        mask_u8=np.asarray(mask, dtype=np.uint8),
        meta={"preset": "marker", "strength": strength},
    )


def _preset_burn(image_u8: np.ndarray, rng: np.random.Generator) -> PresetResult:
    """Burn/heat mark: larger organic dark region with texture."""

    img = _as_u8_image(image_u8)
    h, w = int(img.shape[0]), int(img.shape[1])

    base_res = (max(2, h // 72), max(2, w // 72))
    noise = fractal_perlin_noise_2d((h, w), base_res, rng=rng, octaves=3, persistence=0.6)
    thr = float(rng.uniform(0.52, 0.75))
    mask = ((noise >= thr).astype(np.uint8) * 255).astype(np.uint8)

    try:
        import cv2  # type: ignore
    except Exception:
        cv2 = None
    if cv2 is not None:
        blur = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, ksize=(0, 0), sigmaX=2.2)
        mask = ((blur >= 0.45).astype(np.uint8) * 255).astype(np.uint8)

    strength = float(rng.uniform(0.5, 1.0))
    overlay = _apply_tint(img, mask, rng=rng, strength=strength, mode="darken")
    return PresetResult(
        overlay_u8=np.asarray(overlay, dtype=np.uint8),
        mask_u8=np.asarray(mask, dtype=np.uint8),
        meta={"preset": "burn", "threshold": thr, "strength": strength},
    )


def _preset_bubble(image_u8: np.ndarray, rng: np.random.Generator) -> PresetResult:
    """Bubble-like defect: bright circular blobs (sometimes with dark core)."""

    img = _as_u8_image(image_u8)
    h, w = int(img.shape[0]), int(img.shape[1])

    r = max(2, int(round(min(h, w) * float(rng.uniform(0.015, 0.06)))))
    mask = random_ellipse_mask(
        (h, w),
        rng=rng,
        num_ellipses=int(rng.integers(1, 4)),
        axis_range=(max(2, r // 2), max(3, r)),
        blur_sigma=float(rng.uniform(0.8, 2.2)),
        threshold=0.45,
    )
    strength = float(rng.uniform(0.35, 1.0))
    mode = "brighten" if float(rng.uniform(0.0, 1.0)) < 0.7 else "darken"
    overlay = _apply_tint(img, mask, rng=rng, strength=strength, mode=mode)
    return PresetResult(
        overlay_u8=np.asarray(overlay, dtype=np.uint8),
        mask_u8=np.asarray(mask, dtype=np.uint8),
        meta={"preset": "bubble", "strength": strength, "mode": mode},
    )


def _preset_fiber(image_u8: np.ndarray, rng: np.random.Generator) -> PresetResult:
    """Fibers/hairs: multiple thin curved scratches."""

    img = _as_u8_image(image_u8)
    h, w = int(img.shape[0]), int(img.shape[1])

    spec = ScratchSpec(
        thickness=1,
        length_fraction=float(rng.uniform(0.35, 0.85)),
        jitter=float(rng.uniform(0.02, 0.12)),
        blur_sigma=float(rng.uniform(0.2, 0.9)),
    )
    mask = random_curve_scratch_mask(
        (h, w),
        rng=rng,
        num_scratches=int(rng.integers(3, 10)),
        spec=spec,
        curvature=float(rng.uniform(0.15, 0.35)),
    )
    strength = float(rng.uniform(0.35, 1.0))
    overlay = _apply_tint(img, mask, rng=rng, strength=strength, mode="darken")
    return PresetResult(
        overlay_u8=np.asarray(overlay, dtype=np.uint8),
        mask_u8=np.asarray(mask, dtype=np.uint8),
        meta={"preset": "fiber", "strength": strength},
    )


def _preset_wrinkle(image_u8: np.ndarray, rng: np.random.Generator) -> PresetResult:
    """Wrinkle/crease: wavy stroke across the surface."""

    import cv2  # local import

    img = _as_u8_image(image_u8)
    h, w = int(img.shape[0]), int(img.shape[1])

    mask = np.zeros((h, w), dtype=np.uint8)
    n = int(rng.integers(1, 3))
    horizontal = bool(rng.integers(0, 2))
    for _ in range(n):
        if horizontal:
            y0 = float(rng.uniform(0.1 * h, 0.9 * h))
            amp = float(rng.uniform(0.01, 0.05)) * float(h)
            freq = float(rng.uniform(1.5, 3.5)) / float(max(1, w))
            phase = float(rng.uniform(0.0, 2.0 * np.pi))
            xs = np.linspace(0.0, float(w - 1), num=max(32, w // 3), dtype=np.float32)
            ys = y0 + amp * np.sin(2.0 * np.pi * freq * xs + phase)
        else:
            x0 = float(rng.uniform(0.1 * w, 0.9 * w))
            amp = float(rng.uniform(0.01, 0.05)) * float(w)
            freq = float(rng.uniform(1.5, 3.5)) / float(max(1, h))
            phase = float(rng.uniform(0.0, 2.0 * np.pi))
            ys = np.linspace(0.0, float(h - 1), num=max(32, h // 3), dtype=np.float32)
            xs = x0 + amp * np.sin(2.0 * np.pi * freq * ys + phase)

        pts = np.stack([xs, ys], axis=1)
        pts[:, 0] = np.clip(pts[:, 0], 0.0, float(w - 1))
        pts[:, 1] = np.clip(pts[:, 1], 0.0, float(h - 1))
        pts_i = pts.astype(np.int32).reshape(-1, 1, 2)
        thick = int(rng.integers(2, max(3, min(h, w) // 20)))
        cv2.polylines(mask, [pts_i], isClosed=False, color=255, thickness=thick, lineType=cv2.LINE_AA)

    sigma = float(rng.uniform(0.6, 2.0))
    blur = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, ksize=(0, 0), sigmaX=sigma)
    mask = ((blur >= 0.25).astype(np.uint8) * 255).astype(np.uint8)

    strength = float(rng.uniform(0.35, 1.0))
    overlay = _apply_tint(img, mask, rng=rng, strength=strength, mode="darken")
    return PresetResult(
        overlay_u8=np.asarray(overlay, dtype=np.uint8),
        mask_u8=np.asarray(mask, dtype=np.uint8),
        meta={"preset": "wrinkle", "strength": strength, "horizontal": horizontal},
    )


def _preset_texture(image_u8: np.ndarray, rng: np.random.Generator) -> PresetResult:
    """Texture injection: sample a crop from the same image and blend via an organic mask."""

    import cv2  # local import

    img = _as_u8_image(image_u8)
    h, w = int(img.shape[0]), int(img.shape[1])

    # Sample a random crop and resize back to full image size.
    crop_frac = float(rng.uniform(0.25, 0.8))
    ch = max(2, int(round(crop_frac * h)))
    cw = max(2, int(round(crop_frac * w)))
    y0 = int(rng.integers(0, max(1, h - ch + 1)))
    x0 = int(rng.integers(0, max(1, w - cw + 1)))
    crop = np.asarray(img[y0 : y0 + ch, x0 : x0 + cw], dtype=np.uint8)
    overlay = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.uint8)

    # Light color jitter for "foreign" feel.
    strength = float(rng.uniform(0.25, 0.75))
    jitter = float(rng.uniform(-0.15, 0.15))
    ov_f = overlay.astype(np.float32) * (1.0 + jitter) + float(rng.uniform(-12.0, 12.0)) * strength
    overlay = np.clip(ov_f, 0.0, 255.0).astype(np.uint8)

    # Organic Perlin mask.
    base_res = (max(2, h // 64), max(2, w // 64))
    noise = fractal_perlin_noise_2d((h, w), base_res, rng=rng, octaves=4, persistence=0.55)
    thr = float(rng.uniform(0.55, 0.78))
    mask = ((noise >= thr).astype(np.uint8) * 255).astype(np.uint8)

    if cv2 is not None:
        blur = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, ksize=(0, 0), sigmaX=2.0)
        mask = ((blur >= 0.5).astype(np.uint8) * 255).astype(np.uint8)

    return PresetResult(
        overlay_u8=np.asarray(overlay, dtype=np.uint8),
        mask_u8=np.asarray(mask, dtype=np.uint8),
        meta={"preset": "texture", "threshold": thr, "strength": strength, "crop_frac": crop_frac},
    )


def _preset_edge_wear(image_u8: np.ndarray, rng: np.random.Generator) -> PresetResult:
    """Edge wear: defects concentrated near image borders."""

    img = _as_u8_image(image_u8)
    h, w = int(img.shape[0]), int(img.shape[1])

    mask = random_edge_band_mask(
        (h, w),
        rng=rng,
        width_fraction_range=(0.04, 0.16),
        irregularity=float(rng.uniform(0.2, 0.8)),
        blur_sigma=float(rng.uniform(0.0, 1.4)),
        threshold=0.5,
    )
    strength = float(rng.uniform(0.35, 1.0))
    overlay = _apply_tint(img, mask, rng=rng, strength=strength, mode="darken")
    return PresetResult(
        overlay_u8=np.asarray(overlay, dtype=np.uint8),
        mask_u8=np.asarray(mask, dtype=np.uint8),
        meta={"preset": "edge_wear", "strength": strength},
    )


_PRESETS: dict[str, PresetFn] = {
    "scratch": _preset_scratch,
    "stain": _preset_stain,
    "pit": _preset_pit,
    "glare": _preset_glare,
    "rust": _preset_rust,
    "oil": _preset_oil,
    "crack": _preset_crack,
    "brush": _preset_brush,
    "spatter": _preset_spatter,
    "tape": _preset_tape,
    "marker": _preset_marker,
    "burn": _preset_burn,
    "bubble": _preset_bubble,
    "fiber": _preset_fiber,
    "wrinkle": _preset_wrinkle,
    "texture": _preset_texture,
    "edge_wear": _preset_edge_wear,
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


def make_preset_mixture(
    names: list[str] | tuple[str, ...],
    *,
    weights: list[float] | tuple[float, ...] | None = None,
) -> PresetFn:
    """Create a preset function that samples from multiple presets.

    Notes
    -----
    This is useful for industrial synthetic datasets: instead of a single defect
    type, each sample can draw from a *mixture* of defect generators.
    """

    items = [str(n).strip().lower() for n in list(names)]
    if not items:
        raise ValueError("names must be non-empty")

    fns = [make_preset(n) for n in items]

    p = None
    if weights is not None:
        w = np.asarray(list(weights), dtype=np.float64).reshape(-1)
        if int(w.shape[0]) != len(items):
            raise ValueError("weights length must match names length")
        if np.any(w < 0):
            raise ValueError("weights must be non-negative")
        s = float(np.sum(w))
        if s <= 0:
            raise ValueError("weights must sum to > 0")
        p = (w / s).astype(np.float64, copy=False)

    def _mixed(image_u8: np.ndarray, rng: np.random.Generator) -> PresetResult:
        if p is None:
            idx = int(rng.integers(0, len(fns)))
        else:
            idx = int(rng.choice(len(fns), p=p))

        chosen = items[idx]
        out = fns[idx](image_u8, rng)
        meta = dict(out.meta)
        # Ensure the active preset name is always present/stable.
        meta["preset"] = str(meta.get("preset", chosen))
        meta["preset_mixture"] = list(items)
        return PresetResult(
            overlay_u8=np.asarray(out.overlay_u8, dtype=np.uint8),
            mask_u8=np.asarray(out.mask_u8, dtype=np.uint8),
            meta=meta,
        )

    return _mixed


__all__ = [
    "PresetResult",
    "PresetFn",
    "get_preset_names",
    "make_preset",
    "make_preset_mixture",
]
