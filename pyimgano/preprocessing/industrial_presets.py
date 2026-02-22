from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray


def _require_u8(image: NDArray) -> NDArray:
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        raise TypeError(f"Expected uint8 image input, got dtype={arr.dtype}")
    return arr


def gray_world_white_balance(image: NDArray, *, eps: float = 1e-6) -> NDArray:
    """Apply gray-world white balance.

    Assumes input is RGB/BGR uint8; channel order does not matter for the math.
    """

    img = _require_u8(image)
    if img.ndim == 2:
        return img
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected image shape (H,W,3), got {img.shape}")

    img_f = img.astype(np.float32)
    channel_means = img_f.mean(axis=(0, 1))
    gray_mean = float(channel_means.mean())
    gains = gray_mean / np.maximum(channel_means, float(eps))

    out = img_f * gains.reshape(1, 1, 3)
    out = np.clip(out, 0.0, 255.0)
    return out.astype(np.uint8)


def max_rgb_white_balance(image: NDArray, *, eps: float = 1e-6) -> NDArray:
    """Apply max-RGB white balance (scale channels so max values match)."""

    img = _require_u8(image)
    if img.ndim == 2:
        return img
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected image shape (H,W,3), got {img.shape}")

    img_f = img.astype(np.float32)
    channel_max = img_f.reshape(-1, 3).max(axis=0)
    target = float(channel_max.max())
    gains = target / np.maximum(channel_max, float(eps))

    out = img_f * gains.reshape(1, 1, 3)
    out = np.clip(out, 0.0, 255.0)
    return out.astype(np.uint8)


def homomorphic_filter(
    image: NDArray,
    *,
    cutoff: float = 0.5,
    gamma_low: float = 0.7,
    gamma_high: float = 1.5,
    c: float = 1.0,
    eps: float = 1e-6,
    per_channel: bool = True,
) -> NDArray:
    """Illumination normalization via a simple homomorphic filter.

    Parameters
    ----------
    cutoff:
        Relative cutoff in (0, 1]. Smaller emphasizes high frequencies more.
    gamma_low/gamma_high:
        Low/high frequency gains.
    c:
        Sharpness of the transition.
    per_channel:
        When True, filter each channel independently; otherwise filter intensity.
    """

    img = _require_u8(image)
    if img.ndim not in (2, 3):
        raise ValueError(f"Expected 2D or 3D image, got shape {img.shape}")
    if img.ndim == 3 and img.shape[2] != 3:
        raise ValueError(f"Expected image shape (H,W,3), got {img.shape}")

    cutoff_f = float(cutoff)
    if not (0.0 < cutoff_f <= 1.0):
        raise ValueError(f"cutoff must be in (0,1], got {cutoff}")

    def _filter_channel(channel: NDArray, *, d0: float) -> NDArray:
        ch = channel.astype(np.float32) / 255.0
        ch = np.maximum(ch, float(eps))
        log_im = np.log(ch)

        fft = np.fft.fft2(log_im)
        fft_shift = np.fft.fftshift(fft)

        h, w = int(ch.shape[0]), int(ch.shape[1])
        y, x = np.ogrid[:h, :w]
        cy, cx = h // 2, w // 2
        dist2 = (y - cy) ** 2 + (x - cx) ** 2
        d2 = dist2 / max(float(d0) ** 2, float(eps))
        hpf = (float(gamma_high) - float(gamma_low)) * (1.0 - np.exp(-float(c) * d2)) + float(
            gamma_low
        )

        filtered = fft_shift * hpf
        inv = np.fft.ifft2(np.fft.ifftshift(filtered))
        inv_real = np.real(inv)
        exp_im = np.exp(inv_real)
        exp_im = np.clip(exp_im, 0.0, 1.0)
        return (exp_im * 255.0).astype(np.uint8)

    h, w = int(img.shape[0]), int(img.shape[1])
    d0 = cutoff_f * (max(h, w) / 2.0)

    if img.ndim == 2:
        return _filter_channel(img, d0=d0)

    if per_channel:
        out = np.stack([_filter_channel(img[..., i], d0=d0) for i in range(3)], axis=2)
        return out

    # Intensity-based variant: filter luminance then re-scale RGB channels proportionally.
    img_f = img.astype(np.float32)
    intensity = img_f.mean(axis=2).astype(np.uint8)
    intensity_f = _filter_channel(intensity, d0=d0).astype(np.float32)
    denom = np.maximum(img_f.mean(axis=2, keepdims=True), float(eps))
    out = img_f * (intensity_f[..., None] / denom)
    out = np.clip(out, 0.0, 255.0)
    return out.astype(np.uint8)

