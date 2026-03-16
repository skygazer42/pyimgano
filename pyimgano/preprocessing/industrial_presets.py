from __future__ import annotations

from dataclasses import dataclass

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


@dataclass(frozen=True)
class IlluminationContrastKnobs:
    """Opt-in illumination / contrast preprocessing knobs (industrial-friendly).

    Notes
    -----
    - This is intentionally conservative and `uint8`-preserving.
    - The intended workflow is: apply these knobs *before* feature extraction or
      patch embedding to reduce camera drift / lighting changes.
    """

    white_balance: str = "none"  # none|gray_world|max_rgb
    homomorphic: bool = False
    homomorphic_cutoff: float = 0.5
    homomorphic_gamma_low: float = 0.7
    homomorphic_gamma_high: float = 1.5
    homomorphic_c: float = 1.0
    homomorphic_per_channel: bool = True

    clahe: bool = False
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: tuple[int, int] = (8, 8)

    gamma: float | None = None
    contrast_stretch: bool = False
    contrast_lower_percentile: float = 2.0
    contrast_upper_percentile: float = 98.0


def _clahe_u8(
    image: NDArray,
    *,
    clip_limit: float,
    tile_grid_size: tuple[int, int],
) -> NDArray:
    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "opencv-python is required for CLAHE.\n"
            "Install it via:\n  pip install 'opencv-python'\n"
            f"Original error: {exc}"
        ) from exc

    img = _require_u8(image)
    tgs = (int(tile_grid_size[0]), int(tile_grid_size[1]))
    if tgs[0] <= 0 or tgs[1] <= 0:
        raise ValueError(f"tile_grid_size must be positive ints, got {tile_grid_size}")
    clip = float(clip_limit)
    if clip <= 0:
        raise ValueError(f"clip_limit must be > 0, got {clip_limit}")

    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tgs)
    if img.ndim == 2:
        return np.asarray(clahe.apply(img), dtype=np.uint8)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected image shape (H,W,3) or (H,W), got {img.shape}")

    # Channel-order independent variant:
    # apply CLAHE on intensity, then re-scale RGB channels proportionally.
    img_f = img.astype(np.float32)
    intensity = np.mean(img_f, axis=2)
    intensity_u8 = np.clip(intensity, 0.0, 255.0).astype(np.uint8)
    eq_u8 = np.asarray(clahe.apply(intensity_u8), dtype=np.uint8).astype(np.float32)

    denom = np.maximum(intensity[..., None], 1.0)
    out = img_f * (eq_u8[..., None] / denom)
    out = np.clip(out, 0.0, 255.0)
    return out.astype(np.uint8)


def apply_illumination_contrast(
    image: NDArray,
    *,
    knobs: IlluminationContrastKnobs | None = None,
    white_balance: str | None = None,
    homomorphic: bool | None = None,
    clahe: bool | None = None,
    gamma: float | None = None,
    contrast_stretch: bool | None = None,
) -> NDArray:
    """Apply an opt-in illumination/contrast preprocessing chain.

    Parameters
    ----------
    knobs:
        Optional config object. When provided, individual keyword arguments override it.
    white_balance:
        "none" (default), "gray_world", or "max_rgb".
    homomorphic:
        Enable homomorphic filtering for illumination normalization.
    clahe:
        Enable CLAHE for local contrast enhancement.
    gamma:
        Optional gamma correction. Use <1.0 to brighten, >1.0 to darken.
    contrast_stretch:
        Enable percentile-based contrast stretching.
    """

    k = knobs or IlluminationContrastKnobs()
    wb = str(white_balance if white_balance is not None else k.white_balance).strip().lower()
    hm = bool(homomorphic) if homomorphic is not None else bool(k.homomorphic)
    ch = bool(clahe) if clahe is not None else bool(k.clahe)
    cs = bool(contrast_stretch) if contrast_stretch is not None else bool(k.contrast_stretch)

    out = _require_u8(image)

    if wb in ("gray_world", "gray-world", "grayworld"):
        out = gray_world_white_balance(out)
    elif wb in ("max_rgb", "max-rgb", "maxrgb"):
        out = max_rgb_white_balance(out)
    elif wb not in ("none", ""):
        raise ValueError(
            "white_balance must be one of: none, gray_world, max_rgb. " f"Got: {wb!r}."
        )

    if hm:
        out = homomorphic_filter(
            out,
            cutoff=float(k.homomorphic_cutoff),
            gamma_low=float(k.homomorphic_gamma_low),
            gamma_high=float(k.homomorphic_gamma_high),
            c=float(k.homomorphic_c),
            per_channel=bool(k.homomorphic_per_channel),
        )

    if ch:
        out = _clahe_u8(
            out,
            clip_limit=float(k.clahe_clip_limit),
            tile_grid_size=tuple(k.clahe_tile_grid_size),
        )

    # Optional contrast shaping.
    gamma_v = k.gamma if gamma is None else gamma
    if gamma_v is not None:
        gv = float(gamma_v)
        if gv <= 0:
            raise ValueError(f"gamma must be > 0, got {gamma_v}")

        from pyimgano.preprocessing.advanced_operations import gamma_correction

        out = np.asarray(gamma_correction(out, gamma=gv), dtype=np.uint8)

    if cs:
        from pyimgano.preprocessing.advanced_operations import contrast_stretching

        out = np.asarray(
            contrast_stretching(
                out,
                lower_percentile=float(k.contrast_lower_percentile),
                upper_percentile=float(k.contrast_upper_percentile),
            ),
            dtype=np.uint8,
        )

    return np.asarray(out, dtype=np.uint8)


def shading_correction(
    image: NDArray,
    *,
    radius: int = 50,
    clahe_clip_limit: float = 2.0,
    clahe_tile_grid_size: tuple[int, int] = (8, 8),
) -> NDArray:
    """Shading correction preset: rolling-ball background + CLAHE.

    This is intended for industrial surface images where slow-varying illumination
    changes dominate the signal.

    Notes
    -----
    - Uses a morphological opening as a rolling-ball approximation.
    - Preserves input shape (grayscale stays grayscale, color stays color).
    - Output is uint8.
    """

    img = _require_u8(image)

    from pyimgano.preprocessing.background import estimate_background_rolling_ball

    r = int(radius)
    if r < 0:
        raise ValueError(f"radius must be >= 0, got {radius}")

    if img.ndim == 2:
        bg = estimate_background_rolling_ball(img, radius=r)
        bias = int(np.round(float(np.median(bg))))
        corrected = img.astype(np.int16) - bg.astype(np.int16) + bias
        corrected_u8 = np.clip(corrected, 0, 255).astype(np.uint8)
        return _clahe_u8(
            corrected_u8,
            clip_limit=float(clahe_clip_limit),
            tile_grid_size=tuple(clahe_tile_grid_size),
        )

    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected image shape (H,W) or (H,W,3), got {img.shape}")

    chans: list[np.ndarray] = []
    for c in range(3):
        ch = img[..., c]
        bg = estimate_background_rolling_ball(ch, radius=r)
        bias = int(np.round(float(np.median(bg))))
        corrected = ch.astype(np.int16) - bg.astype(np.int16) + bias
        chans.append(np.clip(corrected, 0, 255).astype(np.uint8))

    corrected_u8 = np.stack(chans, axis=2)
    return _clahe_u8(
        corrected_u8,
        clip_limit=float(clahe_clip_limit),
        tile_grid_size=tuple(clahe_tile_grid_size),
    )


def retinex_illumination_normalization(
    image: NDArray,
    *,
    sigmas: tuple[float, ...] = (15.0, 80.0, 250.0),
    clip_percentiles: tuple[float, float] = (1.0, 99.0),
) -> NDArray:
    """Retinex illumination normalization preset (MSRCR-lite).

    This is a pragmatic, `uint8`-preserving illumination normalization step for
    industrial pipelines.
    """

    img = _require_u8(image)
    from pyimgano.preprocessing.retinex import msrcr_lite

    return np.asarray(
        msrcr_lite(img, sigmas=tuple(float(s) for s in sigmas), clip_percentiles=clip_percentiles),
        dtype=np.uint8,
    )


def defect_amplification(
    image: NDArray,
    *,
    tophat_ksize: tuple[int, int] = (15, 15),
    edge_method: str = "sobel",
    tophat_weight: float = 1.0,
    edge_weight: float = 0.5,
    canny_threshold1: int = 50,
    canny_threshold2: int = 150,
) -> NDArray:
    """Defect amplification preset: top-hat + edges.

    This highlights small bright defects and sharp boundaries. Output is grayscale uint8.
    """

    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "opencv-python is required for defect amplification.\n"
            "Install it via:\n  pip install 'opencv-python'\n"
            f"Original error: {exc}"
        ) from exc

    img = _require_u8(image)
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif img.ndim == 2:
        gray = img
    else:
        raise ValueError(f"Expected image shape (H,W) or (H,W,3), got {img.shape}")

    k0, k1 = int(tophat_ksize[0]), int(tophat_ksize[1])
    if k0 <= 0 or k1 <= 0:
        raise ValueError(f"tophat_ksize must be positive ints, got {tophat_ksize}")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k0, k1))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)

    em = str(edge_method).strip().lower()
    if em == "sobel":
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        mmax = float(np.max(mag)) if mag.size else 0.0
        if mmax > 0:
            edges = np.clip(mag * (255.0 / mmax), 0.0, 255.0).astype(np.uint8)
        else:
            edges = np.zeros_like(gray, dtype=np.uint8)
    elif em == "canny":
        edges = cv2.Canny(gray, int(canny_threshold1), int(canny_threshold2))
    else:
        raise ValueError("edge_method must be one of: sobel, canny")

    out = float(tophat_weight) * tophat.astype(np.float32) + float(edge_weight) * edges.astype(
        np.float32
    )
    return np.clip(out, 0.0, 255.0).astype(np.uint8)
