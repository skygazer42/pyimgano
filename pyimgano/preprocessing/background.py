from __future__ import annotations

"""Background estimation / subtraction helpers (industrial-friendly).

The classic "rolling ball" background subtraction can be approximated well for
many industrial surface inspection tasks using a morphological opening with a
large, disk-like structuring element.

This module provides a small, uint8-first implementation built on OpenCV.
"""

import numpy as np
from numpy.typing import NDArray


def _to_gray_u8(image: NDArray) -> NDArray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        return arr.astype(np.uint8, copy=False)
    if arr.ndim == 3 and arr.shape[2] == 3:
        try:
            import cv2  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "opencv-python is required for rolling-ball background subtraction.\n"
                "Install it via:\n  pip install 'opencv-python'\n"
                f"Original error: {exc}"
            ) from exc
        return cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY).astype(np.uint8, copy=False)
    raise ValueError(f"Expected (H,W) or (H,W,3) image, got {arr.shape}")


def estimate_background_rolling_ball(image: NDArray, *, radius: int = 50) -> NDArray:
    """Estimate a smooth background via a rolling-ball approximation."""

    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "opencv-python is required for rolling-ball background subtraction.\n"
            "Install it via:\n  pip install 'opencv-python'\n"
            f"Original error: {exc}"
        ) from exc

    gray = _to_gray_u8(image)
    r = int(radius)
    if r <= 0:
        return gray.copy()

    k = 2 * r + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(k), int(k)))
    bg = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    return np.asarray(bg, dtype=np.uint8)


def subtract_background_rolling_ball(
    image: NDArray,
    *,
    radius: int = 50,
    light_background: bool = False,
) -> NDArray:
    """Subtract an estimated background (rolling-ball approximation).

    Parameters
    ----------
    image:
        Grayscale or color image (uint8 recommended).
    radius:
        Structuring element radius. Larger values remove slower-varying shading.
    light_background:
        - False: return `image - background` (highlights bright defects)
        - True:  return `background - image` (highlights dark defects)
    """

    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "opencv-python is required for rolling-ball background subtraction.\n"
            "Install it via:\n  pip install 'opencv-python'\n"
            f"Original error: {exc}"
        ) from exc

    gray = _to_gray_u8(image)
    bg = estimate_background_rolling_ball(gray, radius=int(radius))
    if bool(light_background):
        out = cv2.subtract(bg, gray)
    else:
        out = cv2.subtract(gray, bg)
    return np.asarray(out, dtype=np.uint8)
