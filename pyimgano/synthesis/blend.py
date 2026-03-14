from __future__ import annotations

from typing import Literal, Optional

import numpy as np

from .masks import ensure_u8_mask


def _as_u8_image(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        raise ValueError(f"Expected uint8 image, got dtype={arr.dtype}")
    if arr.ndim not in (2, 3):
        raise ValueError(f"Expected grayscale (H,W) or BGR/RGB (H,W,3), got {arr.shape}")
    if arr.ndim == 3 and arr.shape[2] != 3:
        raise ValueError(f"Expected grayscale (H,W) or BGR/RGB (H,W,3), got {arr.shape}")
    return arr


def alpha_blend(
    base_u8: np.ndarray,
    overlay_u8: np.ndarray,
    alpha_mask: np.ndarray,
    *,
    alpha: float = 1.0,
) -> np.ndarray:
    """Alpha blend overlay onto base using a mask.

    Parameters
    ----------
    base_u8 / overlay_u8:
        uint8 images of the same shape.
    alpha_mask:
        Binary or continuous mask. Non-zero means "apply overlay".
    alpha:
        Global strength multiplier in [0,1].
    """

    base = _as_u8_image(base_u8)
    ov = _as_u8_image(overlay_u8)
    if base.shape != ov.shape:
        raise ValueError(f"base and overlay must have same shape, got {base.shape} vs {ov.shape}")

    m = np.asarray(alpha_mask)
    if m.shape != base.shape[:2]:
        raise ValueError(f"alpha_mask must have shape (H,W), got {m.shape} for image {base.shape}")

    m_f = m.astype(np.float32)
    if m_f.dtype != np.float32:
        m_f = m_f.astype(np.float32)
    if float(np.max(m_f)) > 1.0:
        m_f = m_f / 255.0
    m_f = np.clip(m_f, 0.0, 1.0) * float(np.clip(alpha, 0.0, 1.0))

    b = base.astype(np.float32)
    o = ov.astype(np.float32)
    if base.ndim == 3:
        out = b * (1.0 - m_f[..., None]) + o * m_f[..., None]
    else:
        out = b * (1.0 - m_f) + o * m_f
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


_PoissonMode = Literal["normal", "mixed"]


def poisson_blend(
    base_u8: np.ndarray,
    overlay_u8: np.ndarray,
    mask_u8: np.ndarray,
    *,
    center_xy: Optional[tuple[int, int]] = None,
    mode: _PoissonMode = "normal",
) -> np.ndarray:
    """Poisson blend overlay onto base via OpenCV seamlessClone.

    Notes
    -----
    - OpenCV expects 3-channel images in BGR order.
    - When the mask is empty, we fall back to returning the base image.
    """

    import cv2  # local import

    base = _as_u8_image(base_u8)
    ov = _as_u8_image(overlay_u8)
    if base.shape != ov.shape:
        raise ValueError(f"base and overlay must have same shape, got {base.shape} vs {ov.shape}")

    mask = ensure_u8_mask(mask_u8, shape_hw=base.shape[:2])
    if int(np.sum(mask > 0)) == 0:
        return np.asarray(base, dtype=np.uint8)

    if base.ndim == 2:
        base3 = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
        ov3 = cv2.cvtColor(ov, cv2.COLOR_GRAY2BGR)
    else:
        base3 = base
        ov3 = ov

    h, w = int(base.shape[0]), int(base.shape[1])
    if center_xy is None:
        center_xy = (w // 2, h // 2)

    flag = cv2.NORMAL_CLONE if mode == "normal" else cv2.MIXED_CLONE
    try:
        out = cv2.seamlessClone(ov3, base3, mask, center_xy, flag)
    except cv2.error:
        # seamlessClone can fail on some degenerate masks; fall back to alpha.
        out = alpha_blend(base, ov, mask, alpha=1.0)

    if base.ndim == 2:
        out = cv2.cvtColor(np.asarray(out, dtype=np.uint8), cv2.COLOR_BGR2GRAY)
    return np.asarray(out, dtype=np.uint8)
