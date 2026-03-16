from __future__ import annotations

from typing import Literal, Optional

import numpy as np

IlluminationMode = Literal["gradient", "vignette"]


def _as_u8_image(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        raise ValueError(f"Expected uint8 image, got dtype={arr.dtype}")
    if arr.ndim not in (2, 3):
        raise ValueError(f"Expected grayscale (H,W) or color (H,W,3) image, got {arr.shape}")
    if arr.ndim == 3 and arr.shape[2] != 3:
        raise ValueError(f"Expected grayscale (H,W) or color (H,W,3) image, got {arr.shape}")
    return arr


def apply_illumination_shift(
    image_u8: np.ndarray,
    *,
    rng: np.random.Generator,
    mode: Optional[IlluminationMode] = None,
    strength: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Apply a global illumination shift (gradient/vignette) and return (overlay, mask, meta).

    Notes
    -----
    - This is intentionally lightweight and deterministic under `rng`.
    - The output mask is **dense** (full image) because illumination shifts affect most pixels.
    """

    img = _as_u8_image(image_u8)
    h, w = int(img.shape[0]), int(img.shape[1])
    if h <= 0 or w <= 0:
        empty = np.zeros((max(0, h), max(0, w)), dtype=np.uint8)
        return np.asarray(img, dtype=np.uint8), empty, {"skipped": True, "reason": "empty_image"}

    chosen_mode: IlluminationMode
    if mode is None:
        chosen_mode = "vignette" if float(rng.uniform(0.0, 1.0)) < 0.5 else "gradient"
    else:
        chosen_mode = mode

    # Strength is normalized ~[0,1]. Larger => more severe illumination change.
    s = float(rng.uniform(0.25, 0.85) if strength is None else strength)
    s = float(np.clip(s, 0.0, 1.0))

    yy, xx = np.meshgrid(
        np.linspace(0.0, 1.0, num=h, dtype=np.float32),
        np.linspace(0.0, 1.0, num=w, dtype=np.float32),
        indexing="ij",
    )

    if chosen_mode == "gradient":
        # Directional lighting gradient (sensor/illumination drift).
        theta = float(rng.uniform(0.0, 2.0 * np.pi))
        dx = float(np.cos(theta))
        dy = float(np.sin(theta))
        # Center and normalize to roughly [-1,1].
        t = (xx - 0.5) * dx + (yy - 0.5) * dy
        t = (t / (np.max(np.abs(t)) + 1e-6)).astype(np.float32, copy=False)
        # Multiplicative gain around 1.0.
        gain = (1.0 + (2.0 * t) * (0.35 * s)).astype(np.float32)
        meta: dict[str, object] = {"mode": "gradient", "strength": float(s), "theta": float(theta)}
    elif chosen_mode == "vignette":
        # Darken edges (classic vignetting / lens shading).
        cx = float(rng.uniform(0.35, 0.65))
        cy = float(rng.uniform(0.35, 0.65))
        r2 = ((xx - cx) ** 2 + (yy - cy) ** 2).astype(np.float32, copy=False)
        # r2 is ~[0, ~0.5] for typical images. Scale to [0,1] and apply strength.
        r2 = (r2 / (float(np.max(r2)) + 1e-6)).astype(np.float32, copy=False)
        gain = (1.0 - (r2**1.15) * (0.55 * s)).astype(np.float32)
        meta = {"mode": "vignette", "strength": float(s), "center_xy_norm": (float(cx), float(cy))}
    else:  # pragma: no cover - type safety
        raise ValueError(f"Unknown mode: {chosen_mode!r}")

    gain = np.clip(gain, 0.3, 1.7).astype(np.float32, copy=False)

    out = img.astype(np.float32)
    if out.ndim == 3:
        out = out * gain[..., None]
    else:
        out = out * gain

    overlay = np.clip(out, 0.0, 255.0).astype(np.uint8)
    mask = np.full((h, w), 255, dtype=np.uint8)
    return overlay, mask, meta


__all__ = ["IlluminationMode", "apply_illumination_shift"]
