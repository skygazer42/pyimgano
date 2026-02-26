from __future__ import annotations

"""Perona–Malik anisotropic diffusion (edge-preserving smoothing).

This is a small, dependency-light implementation intended for industrial
preprocessing pipelines.
"""

import numpy as np
from numpy.typing import NDArray


def anisotropic_diffusion(
    image: NDArray,
    *,
    niter: int = 10,
    kappa: float = 50.0,
    gamma: float = 0.1,
    option: int = 1,
) -> NDArray:
    """Apply anisotropic diffusion to a grayscale (or color) image.

    Parameters
    ----------
    image:
        Input image. If color (H,W,3), it will be converted to grayscale.
    niter:
        Number of iterations.
    kappa:
        Conduction coefficient. Larger values preserve fewer edges.
    gamma:
        Integration constant. Typical stable range is (0, 0.25].
    option:
        1 uses exponential conduction (Perona–Malik eq. 1),
        2 uses reciprocal conduction (Perona–Malik eq. 2).

    Returns
    -------
    NDArray
        Smoothed uint8 grayscale image.
    """

    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[2] == 3:
        try:
            import cv2  # type: ignore
        except Exception:
            arr = np.mean(arr.astype(np.float32), axis=2)
        else:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    if arr.ndim != 2:
        raise ValueError(f"Expected grayscale image (H,W) or color (H,W,3), got {arr.shape}")

    iters = int(niter)
    if iters < 0:
        raise ValueError(f"niter must be >= 0, got {niter}")
    if iters == 0:
        return arr.astype(np.uint8, copy=True)

    k = float(kappa)
    if k <= 0:
        raise ValueError(f"kappa must be > 0, got {kappa}")

    g = float(gamma)
    if not (0.0 < g <= 0.25):
        raise ValueError(f"gamma must be in (0, 0.25], got {gamma}")

    opt = int(option)
    if opt not in (1, 2):
        raise ValueError(f"option must be 1 or 2, got {option}")

    img = arr.astype(np.float32, copy=False)

    for _ in range(iters):
        # Directional gradients (4-neighborhood).
        nabla_n = np.roll(img, -1, axis=0) - img  # north
        nabla_s = np.roll(img, 1, axis=0) - img   # south
        nabla_e = np.roll(img, -1, axis=1) - img  # east
        nabla_w = np.roll(img, 1, axis=1) - img   # west

        if opt == 1:
            c_n = np.exp(-((nabla_n / k) ** 2))
            c_s = np.exp(-((nabla_s / k) ** 2))
            c_e = np.exp(-((nabla_e / k) ** 2))
            c_w = np.exp(-((nabla_w / k) ** 2))
        else:
            c_n = 1.0 / (1.0 + (nabla_n / k) ** 2)
            c_s = 1.0 / (1.0 + (nabla_s / k) ** 2)
            c_e = 1.0 / (1.0 + (nabla_e / k) ** 2)
            c_w = 1.0 / (1.0 + (nabla_w / k) ** 2)

        img = img + g * (c_n * nabla_n + c_s * nabla_s + c_e * nabla_e + c_w * nabla_w)

    return np.clip(img, 0.0, 255.0).astype(np.uint8)

