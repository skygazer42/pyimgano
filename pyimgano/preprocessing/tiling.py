from __future__ import annotations

"""Tiling helpers for high-resolution preprocessing.

Some industrial images are too large to process efficiently as a single frame,
especially for CPU-heavy operations. This module provides a small helper to:

- split an image into overlapping tiles
- apply a callable per-tile
- blend tiles back together to avoid seams
"""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray


def _weight_window(kind: str, tile_size: int) -> NDArray:
    t = int(tile_size)
    if t <= 0:
        raise ValueError(f"tile_size must be positive, got {tile_size}")

    k = str(kind).strip().lower()
    if k in ("mean", "avg", "average", "flat"):
        return np.ones((t, t), dtype=np.float32)

    if k == "hann":
        w = np.hanning(t).astype(np.float32)
        if float(w.sum()) <= 0:  # pragma: no cover - tiny t edge cases
            w = np.ones((t,), dtype=np.float32)
        # Avoid exact zeros at borders to prevent division-by-zero for boundary pixels.
        w = np.maximum(w, 1e-3)
        return np.outer(w, w).astype(np.float32)

    if k == "gaussian":
        if t == 1:
            return np.ones((1, 1), dtype=np.float32)
        yy, xx = np.mgrid[0:t, 0:t].astype(np.float32)
        cy = (t - 1) / 2.0
        cx = (t - 1) / 2.0
        y = (yy - cy) / max(cy, 1.0)
        x = (xx - cx) / max(cx, 1.0)
        r2 = x * x + y * y
        sigma2 = 0.5 * 0.5
        w = np.exp(-0.5 * r2 / sigma2).astype(np.float32)
        w = np.maximum(w, 1e-3)
        return w

    raise ValueError(f"Unknown blend window: {kind!r}. Choose from: mean, hann, gaussian.")


def _tile_starts(length: int, *, tile_size: int, stride: int) -> list[int]:
    L = int(length)
    t = int(tile_size)
    s = int(stride)
    if L <= t:
        return [0]
    starts = list(range(0, L - t + 1, s))
    last = L - t
    if starts[-1] != last:
        starts.append(last)
    return starts


def tile_apply(
    image: NDArray,
    fn: Callable[[NDArray], NDArray],
    *,
    tile_size: int = 512,
    overlap: int = 64,
    blend: str = "hann",
    pad_mode: str = "edge",
) -> NDArray:
    """Apply `fn` on overlapping tiles and blend back to full resolution.

    Parameters
    ----------
    image:
        Grayscale (H,W) or color (H,W,3) image.
    fn:
        Callable applied to each tile. Must return the same shape as the tile.
    tile_size:
        Tile side length in pixels.
    overlap:
        Overlap in pixels between tiles. Stride = tile_size - overlap.
    blend:
        Blend window: "mean" (flat), "hann", or "gaussian".
    pad_mode:
        NumPy padding mode when the input is smaller than one tile.

    Returns
    -------
    NDArray
        Output image with the same shape/dtype as input (uint8 inputs preserve uint8).
    """

    img = np.asarray(image)
    if img.ndim == 2:
        h, w = int(img.shape[0]), int(img.shape[1])
        channels = None
    elif img.ndim == 3 and img.shape[2] == 3:
        h, w = int(img.shape[0]), int(img.shape[1])
        channels = 3
    else:
        raise ValueError(f"Expected (H,W) or (H,W,3) image, got {img.shape}")

    t = int(tile_size)
    o = int(overlap)
    if t <= 0:
        raise ValueError(f"tile_size must be > 0, got {tile_size}")
    if o < 0:
        raise ValueError(f"overlap must be >= 0, got {overlap}")
    stride = t - o
    if stride <= 0:
        raise ValueError("overlap must be smaller than tile_size")

    # Pad small images so we always operate on full tiles.
    pad_h = max(0, t - h)
    pad_w = max(0, t - w)
    if pad_h or pad_w:
        if channels is None:
            padded = np.pad(img, ((0, pad_h), (0, pad_w)), mode=str(pad_mode))
        else:
            padded = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode=str(pad_mode))
    else:
        padded = img

    ph, pw = int(padded.shape[0]), int(padded.shape[1])
    ys = _tile_starts(ph, tile_size=t, stride=stride)
    xs = _tile_starts(pw, tile_size=t, stride=stride)

    weights = _weight_window(blend, t)
    accum = np.zeros_like(padded, dtype=np.float32)
    wsum = np.zeros((ph, pw), dtype=np.float32)

    for y0 in ys:
        y1 = y0 + t
        for x0 in xs:
            x1 = x0 + t
            tile = padded[y0:y1, x0:x1] if channels is None else padded[y0:y1, x0:x1, :]
            out_tile = np.asarray(fn(np.asarray(tile)))
            if out_tile.shape != tile.shape:
                raise ValueError(
                    f"fn(tile) must return same shape as tile, got {out_tile.shape} vs {tile.shape}"
                )

            out_f = out_tile.astype(np.float32, copy=False)
            if channels is None:
                accum[y0:y1, x0:x1] += out_f * weights
            else:
                accum[y0:y1, x0:x1, :] += out_f * weights[:, :, None]
            wsum[y0:y1, x0:x1] += weights

    denom = np.maximum(wsum, 1e-6)
    if channels is None:
        out = accum / denom
    else:
        out = accum / denom[:, :, None]

    # Crop back to original size.
    if pad_h or pad_w:
        out = out[:h, :w] if channels is None else out[:h, :w, :]

    if img.dtype == np.uint8:
        return np.rint(np.clip(out, 0.0, 255.0)).astype(np.uint8)
    return out.astype(img.dtype, copy=False)
