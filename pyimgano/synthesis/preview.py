from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np


def _as_u8_color(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        raise ValueError(f"Expected uint8 image, got dtype={arr.dtype}")
    if arr.ndim == 2:
        # Convert to 3-channel for preview.
        return np.stack([arr, arr, arr], axis=-1).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[2] == 3:
        return arr
    raise ValueError(f"Expected (H,W) or (H,W,3) image, got {arr.shape}")


def make_preview_grid(
    images: Sequence[np.ndarray],
    *,
    masks: Optional[Sequence[np.ndarray]] = None,
    cols: int = 4,
    pad: int = 2,
    mask_color: tuple[int, int, int] = (0, 0, 255),
    mask_alpha: float = 0.35,
) -> np.ndarray:
    """Build a simple visualization grid (uint8 BGR-ish)."""

    ims = [_as_u8_color(im) for im in images]
    if not ims:
        return np.zeros((0, 0, 3), dtype=np.uint8)

    h0, w0 = int(ims[0].shape[0]), int(ims[0].shape[1])
    for im in ims:
        if im.shape[:2] != (h0, w0):
            raise ValueError("All images must have the same spatial shape for preview grid.")

    ms: list[np.ndarray] | None = None
    if masks is not None:
        ms = [np.asarray(m) for m in masks]
        if len(ms) != len(ims):
            raise ValueError("masks length must match images length")
        for m in ms:
            if m.shape != (h0, w0):
                raise ValueError("All masks must have shape (H,W) matching images")

    c = max(1, int(cols))
    n = len(ims)
    r = int(np.ceil(n / c))

    pad_px = max(0, int(pad))
    out_h = r * h0 + (r - 1) * pad_px
    out_w = c * w0 + (c - 1) * pad_px
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)

    for idx, im in enumerate(ims):
        rr = idx // c
        cc = idx % c
        y0 = rr * (h0 + pad_px)
        x0 = cc * (w0 + pad_px)
        tile = np.asarray(im, dtype=np.uint8)

        if ms is not None:
            m = np.asarray(ms[idx]) > 0
            if np.any(m):
                overlay = tile.astype(np.float32)
                color = np.asarray(mask_color, dtype=np.float32).reshape(1, 1, 3)
                a = float(np.clip(mask_alpha, 0.0, 1.0))
                overlay[m] = overlay[m] * (1.0 - a) + color * a
                tile = np.clip(overlay, 0.0, 255.0).astype(np.uint8)

        canvas[y0 : y0 + h0, x0 : x0 + w0] = tile

    return canvas


def save_preview_grid(
    path: str | Path,
    images: Sequence[np.ndarray],
    *,
    masks: Optional[Sequence[np.ndarray]] = None,
    cols: int = 4,
) -> Path:
    import cv2  # local import

    out = make_preview_grid(images, masks=masks, cols=int(cols))
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(p), out)
    if not ok:
        raise IOError(f"Failed to write preview grid to {p}")
    return p


def show_preview_grid(
    images: Sequence[np.ndarray],
    *,
    masks: Optional[Sequence[np.ndarray]] = None,
    cols: int = 4,
) -> None:
    """Preview grid using matplotlib (optional)."""

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "matplotlib is required for show_preview_grid(). Install it via:\n"
            "  pip install 'matplotlib'\n"
            f"Original error: {exc}"
        ) from exc

    grid = make_preview_grid(images, masks=masks, cols=int(cols))
    # Convert BGR-ish to RGB for display.
    rgb = grid[..., ::-1]
    plt.figure(figsize=(10, 6))
    plt.imshow(rgb)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
