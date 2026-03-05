from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

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


@dataclass(frozen=True)
class RetinexConfig:
    sigmas: tuple[float, ...] = (15.0, 80.0, 250.0)
    clip_percentiles: tuple[float, float] = (1.0, 99.0)


def _normalize_percentile(x: np.ndarray, *, low: float, high: float) -> np.ndarray:
    lo = float(np.percentile(x, low))
    hi = float(np.percentile(x, high))
    denom = hi - lo
    if denom <= 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    out = (x - lo) / denom
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def msrcr_lite(
    image_u8: np.ndarray,
    *,
    sigmas: Sequence[float] = (15.0, 80.0, 250.0),
    clip_percentiles: tuple[float, float] = (1.0, 99.0),
) -> np.ndarray:
    """Multi-Scale Retinex (lite) for illumination normalization.

    This is a pragmatic variant intended for industrial preprocessing:
    - compute multi-scale log-domain reflectance
    - rescale via robust percentiles back to uint8
    """

    import cv2  # local import

    img = _as_u8_image(image_u8)
    sigs = [float(s) for s in sigmas]
    if not sigs:
        raise ValueError("sigmas must be non-empty")
    low_p, high_p = float(clip_percentiles[0]), float(clip_percentiles[1])
    if not (0.0 <= low_p < high_p <= 100.0):
        raise ValueError(
            f"clip_percentiles must satisfy 0<=low<high<=100, got {clip_percentiles!r}"
        )

    x = img.astype(np.float32) + 1.0

    if x.ndim == 2:
        acc = np.zeros_like(x, dtype=np.float32)
        for s in sigs:
            blur = cv2.GaussianBlur(x, ksize=(0, 0), sigmaX=float(s)) + 1.0
            acc = acc + (np.log(x) - np.log(blur))
        acc = acc / float(len(sigs))
        norm = _normalize_percentile(acc, low=low_p, high=high_p)
        return (norm * 255.0).astype(np.uint8)

    # Per-channel, then recombine.
    outs: list[np.ndarray] = []
    for c in range(3):
        ch = x[:, :, c]
        acc = np.zeros_like(ch, dtype=np.float32)
        for s in sigs:
            blur = cv2.GaussianBlur(ch, ksize=(0, 0), sigmaX=float(s)) + 1.0
            acc = acc + (np.log(ch) - np.log(blur))
        acc = acc / float(len(sigs))
        norm = _normalize_percentile(acc, low=low_p, high=high_p)
        outs.append((norm * 255.0).astype(np.uint8))

    out = np.stack(outs, axis=-1)
    return np.asarray(out, dtype=np.uint8)
