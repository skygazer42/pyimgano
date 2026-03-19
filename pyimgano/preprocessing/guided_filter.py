from __future__ import annotations

"""Guided filter (edge-preserving smoothing).

This is a lightweight implementation of the classic guided filter using mean
(box) filtering. It is intended for industrial preprocessing pipelines where
you want to denoise while preserving edges (e.g., surface defects).

We implement the common self-guided variant by default.
"""

from typing import Optional

import numpy as np
from numpy.typing import NDArray


def _to_gray(image: NDArray) -> NDArray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[2] == 3:
        try:
            import cv2  # type: ignore
        except Exception:
            # Fallback: average channels (keeps dependency footprint small).
            return np.mean(arr, axis=2)
        return cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    raise ValueError(f"Expected (H,W) or (H,W,3) image, got {arr.shape}")


def _box_mean(img: NDArray, *, radius: int) -> NDArray:
    """Local mean with a (2r+1)x(2r+1) box window."""

    r = int(radius)
    if r < 0:
        raise ValueError(f"radius must be >= 0, got {radius}")
    if r == 0:
        return np.asarray(img)

    k = 2 * r + 1
    arr = np.asarray(img, dtype=np.float32)

    try:
        import cv2  # type: ignore
    except Exception:
        # NumPy fallback using integral image (replicate padding).
        padded = np.pad(arr, ((r, r), (r, r)), mode="edge")
        ii = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)
        h, w = int(arr.shape[0]), int(arr.shape[1])

        y0 = np.arange(0, h, dtype=np.int64)
        x0 = np.arange(0, w, dtype=np.int64)
        y1 = y0 + k
        x1 = x0 + k

        # Broadcast window sums.
        s = (
            ii[y1[:, None], x1[None, :]]
            - ii[y0[:, None], x1[None, :]]
            - ii[y1[:, None], x0[None, :]]
            + ii[y0[:, None], x0[None, :]]
        )
        return (s / float(k * k)).astype(np.float32, copy=False)

    # OpenCV box filter (fast, stable border handling).
    return cv2.boxFilter(
        arr,
        ddepth=-1,
        ksize=(int(k), int(k)),
        borderType=cv2.BORDER_REPLICATE,
        normalize=True,
    ).astype(np.float32, copy=False)


def guided_filter(
    image: NDArray,
    *,
    guidance: Optional[NDArray] = None,
    radius: int = 4,
    eps: float = 1e-3,
) -> NDArray:
    """Apply a guided filter.

    Parameters
    ----------
    image:
        Input image to be filtered (grayscale or color). For color inputs, we
        currently use a grayscale guidance approximation.
    guidance:
        Optional guidance image. If None, uses `image` (self-guided filter).
    radius:
        Window radius r. Window size is (2r+1)x(2r+1).
    eps:
        Regularization parameter. Larger values produce more smoothing.

    Returns
    -------
    NDArray
        Filtered image. If input is uint8, output is uint8.
    """

    img0 = np.asarray(image)
    g0 = img0 if guidance is None else np.asarray(guidance)

    img = _to_gray(img0)
    g = _to_gray(g0)
    if img.shape != g.shape:
        raise ValueError(
            f"image and guidance must have the same shape, got {img.shape} vs {g.shape}"
        )

    r = int(radius)
    e = float(eps)
    if r < 0:
        raise ValueError(f"radius must be >= 0, got {radius}")
    if e <= 0:
        raise ValueError(f"eps must be > 0, got {eps}")

    # Work in [0,1] if input is uint8 for numerical stability.
    uint8_in = img.dtype == np.uint8
    if uint8_in:
        guidance_f = g.astype(np.float32) / 255.0
        p = img.astype(np.float32) / 255.0
    else:
        guidance_f = g.astype(np.float32)
        p = img.astype(np.float32)

    mean_i = _box_mean(guidance_f, radius=r)
    mean_p = _box_mean(p, radius=r)
    corr_i = _box_mean(guidance_f * guidance_f, radius=r)
    corr_ip = _box_mean(guidance_f * p, radius=r)

    var_i = corr_i - mean_i * mean_i
    cov_ip = corr_ip - mean_i * mean_p

    a = cov_ip / (var_i + e)
    b = mean_p - a * mean_i

    mean_a = _box_mean(a, radius=r)
    mean_b = _box_mean(b, radius=r)

    q = mean_a * guidance_f + mean_b

    if uint8_in:
        out = np.clip(q * 255.0, 0.0, 255.0)
        return np.rint(out).astype(np.uint8)
    return q.astype(np.float32, copy=False)
