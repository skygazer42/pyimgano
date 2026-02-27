from __future__ import annotations

"""Texture/source banks for synthesis-based anomaly injection.

Industrial synthetic anomaly pipelines often paste "foreign" textures into a
normal image (anomalib / CutPaste family). This module provides a lightweight
source bank abstraction without adding new dependencies.
"""

from pathlib import Path
from typing import Any, Sequence

import numpy as np


def _as_u8_color(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr_f = arr.astype(np.float32, copy=False)
        if float(np.nanmax(arr_f)) <= 1.0:
            arr_f = arr_f * 255.0
        arr = np.clip(arr_f, 0.0, 255.0).astype(np.uint8)
    if arr.ndim == 2:
        return np.stack([arr, arr, arr], axis=-1).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[2] == 3:
        return arr.astype(np.uint8, copy=False)
    raise ValueError(f"Expected (H,W) or (H,W,3) image, got {arr.shape}")


def _read_u8_bgr(path: str | Path) -> np.ndarray:
    import cv2  # local import

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return np.asarray(img, dtype=np.uint8)


class TextureSourceBank:
    """A small in-memory bank of source textures for synthesis.

    Parameters
    ----------
    sources:
        A list of source images, either as numpy arrays or as file paths. Paths
        are read via OpenCV (BGR order).
    """

    def __init__(self, sources: Sequence[Any]) -> None:
        items = list(sources)
        if not items:
            raise ValueError("sources must be non-empty")

        imgs: list[np.ndarray] = []
        for s in items:
            if isinstance(s, (str, Path)):
                imgs.append(_read_u8_bgr(s))
            elif isinstance(s, np.ndarray):
                imgs.append(_as_u8_color(s))
            else:
                raise TypeError(f"Unsupported source type: {type(s)}")

        self.sources_u8 = imgs

    def __len__(self) -> int:
        return len(self.sources_u8)

    def sample_overlay(
        self,
        shape_hw: tuple[int, int],
        *,
        rng: np.random.Generator,
        min_crop_frac: float = 0.3,
        max_crop_frac: float = 0.9,
        color_jitter: float = 0.15,
    ) -> np.ndarray:
        """Sample a (H,W,3) overlay by random-cropping a bank image and resizing."""

        import cv2  # local import

        h, w = int(shape_hw[0]), int(shape_hw[1])
        if h <= 0 or w <= 0:
            raise ValueError(f"shape_hw must be positive, got {shape_hw!r}")

        src = self.sources_u8[int(rng.integers(0, len(self.sources_u8)))]
        sh, sw = int(src.shape[0]), int(src.shape[1])
        if sh <= 1 or sw <= 1:
            return np.zeros((h, w, 3), dtype=np.uint8)

        lo = float(np.clip(min_crop_frac, 0.05, 1.0))
        hi = float(np.clip(max_crop_frac, lo, 1.0))
        frac = float(rng.uniform(lo, hi))
        crop_h = max(2, int(round(frac * sh)))
        crop_w = max(2, int(round(frac * sw)))

        y0 = int(rng.integers(0, max(1, sh - crop_h + 1)))
        x0 = int(rng.integers(0, max(1, sw - crop_w + 1)))
        patch = src[y0 : y0 + crop_h, x0 : x0 + crop_w]

        overlay = cv2.resize(patch, (w, h), interpolation=cv2.INTER_LINEAR)
        out = overlay.astype(np.float32)

        cj = float(np.clip(color_jitter, 0.0, 1.0))
        if cj > 0.0:
            # Simple per-channel affine jitter.
            scale = rng.uniform(1.0 - cj, 1.0 + cj, size=(1, 1, 3)).astype(np.float32)
            bias = rng.uniform(-20.0 * cj, 20.0 * cj, size=(1, 1, 3)).astype(np.float32)
            out = out * scale + bias

        return np.clip(out, 0.0, 255.0).astype(np.uint8)

