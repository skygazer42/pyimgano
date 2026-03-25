from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _load_rgb_path_image(path: str | Path) -> np.ndarray:
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")

    if image.ndim == 2:
        image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    elif image.ndim == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError(f"Unsupported image shape from path {path!r}: {image.shape}")

    return np.asarray(image)


def _coerce_single_rgb_image(item: Any) -> np.ndarray:
    if isinstance(item, (str, Path)):
        return _load_rgb_path_image(item)

    arr = np.asarray(item)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, np.newaxis], 3, axis=2)
    elif arr.ndim == 3:
        if arr.shape[-1] in (1, 3, 4):
            pass
        elif arr.shape[0] in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))
        else:
            raise ValueError(f"Unsupported image shape: {arr.shape}")
    else:
        raise ValueError(f"Expected image with 2 or 3 dims, got {arr.shape}")

    if arr.ndim != 3:
        raise ValueError(f"Expected HWC image after normalization, got {arr.shape}")

    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif arr.shape[2] == 4:
        arr = arr[:, :, :3]
    elif arr.shape[2] != 3:
        raise ValueError(f"Unsupported channel count: {arr.shape}")

    return np.asarray(arr)


def coerce_rgb_image_batch(x: Any) -> np.ndarray:
    """Return a `(N, H, W, 3)` numpy batch from paths, arrays, or iterables."""

    if isinstance(x, np.ndarray):
        arr = np.asarray(x)
        if arr.ndim == 4:
            return np.asarray([_coerce_single_rgb_image(arr[i]) for i in range(int(arr.shape[0]))])
        if arr.ndim in (2, 3):
            return np.asarray([_coerce_single_rgb_image(arr)])
        raise ValueError(f"Expected image batch with 2-4 dims, got {arr.shape}")

    if isinstance(x, (str, Path)):
        return np.asarray([_coerce_single_rgb_image(x)])

    items = list(x)
    if not items:
        raise ValueError("Expected at least one image input.")

    return np.asarray([_coerce_single_rgb_image(item) for item in items])
