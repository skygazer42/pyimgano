from __future__ import annotations

from pathlib import Path

import numpy as np


def save_binary_mask(mask_u8: np.ndarray, path: str | Path, *, format: str) -> str:
    """Save a binary mask to disk (png or npy).

    Args:
        mask_u8: 2D uint8 mask, typically with values {0,255}.
        path: Output path.
        format: "png" or "npy".

    Returns:
        The written path as a string.
    """

    fmt = str(format).lower().strip()
    if fmt == "png":
        return save_binary_mask_png(mask_u8, path)
    if fmt == "npy":
        return save_binary_mask_npy(mask_u8, path)
    raise ValueError(f"Unknown mask format: {format!r}. Expected 'png' or 'npy'.")


def save_binary_mask_png(mask_u8: np.ndarray, path: str | Path) -> str:
    from PIL import Image

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    arr = np.asarray(mask_u8, dtype=np.uint8)
    if arr.ndim != 2:
        raise ValueError(f"mask_u8 must be 2D for PNG export, got shape {arr.shape}")

    Image.fromarray(arr, mode="L").save(p)
    return str(p)


def save_binary_mask_npy(mask_u8: np.ndarray, path: str | Path) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    arr = np.asarray(mask_u8, dtype=np.uint8)
    if arr.ndim != 2:
        raise ValueError(f"mask_u8 must be 2D for NPY export, got shape {arr.shape}")

    np.save(p, arr)
    # np.save may append ".npy" when suffix is missing.
    if p.suffix.lower() != ".npy":
        p = p.with_suffix(p.suffix + ".npy")
    return str(p)

