from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray


def _validate_severity(severity: int) -> int:
    s = int(severity)
    if s < 1 or s > 5:
        raise ValueError(f"severity must be in [1,5], got {severity}")
    return s


def _as_u8_image(image: NDArray) -> NDArray:
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        raise ValueError(f"Expected uint8 image, got dtype={arr.dtype}")
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[2] == 3:
        return arr
    raise ValueError(f"Expected grayscale (H,W) or color (H,W,3) image, got {arr.shape}")


def apply_lighting(
    image: NDArray,
    *,
    mask: Optional[NDArray],
    severity: int,
    rng: np.random.Generator,
) -> tuple[NDArray, Optional[NDArray]]:
    """Apply a deterministic lighting/exposure/white-balance corruption."""

    arr = _as_u8_image(image)
    s = _validate_severity(severity)
    strength = float(s) / 5.0

    brightness = float(rng.uniform(-0.15, 0.15) * strength * 255.0)
    contrast = float(1.0 + rng.uniform(-0.25, 0.25) * strength)
    gamma = float(1.0 + rng.uniform(-0.25, 0.25) * strength)
    gamma = max(gamma, 0.1)

    out = arr.astype(np.float32)
    if out.ndim == 3:
        gains = rng.uniform(1.0 - 0.12 * strength, 1.0 + 0.12 * strength, size=(3,)).astype(
            np.float32
        )
        out = out * gains.reshape(1, 1, 3)

    out = out * contrast + brightness
    out = np.clip(out, 0.0, 255.0)
    out = ((out / 255.0) ** gamma) * 255.0
    out_u8 = np.clip(out, 0.0, 255.0).astype(np.uint8)
    return out_u8, mask


def _jpeg_quality_for_severity(severity: int) -> int:
    s = _validate_severity(severity)
    # Coarse mapping chosen to create visible artifacts by severity.
    mapping = {
        1: 90,
        2: 70,
        3: 50,
        4: 30,
        5: 15,
    }
    return int(mapping[s])


def apply_jpeg(
    image: NDArray,
    *,
    mask: Optional[NDArray],
    severity: int,
    rng: np.random.Generator,
) -> tuple[NDArray, Optional[NDArray]]:
    """Apply a deterministic JPEG encode/decode corruption."""

    _ = rng  # reserved for future randomized variants; keep signature consistent
    arr = _as_u8_image(image)
    quality = _jpeg_quality_for_severity(severity)

    try:
        from io import BytesIO
        from PIL import Image
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "Pillow is required for JPEG corruption.\n"
            "Install it via:\n  pip install 'Pillow'\n"
            f"Original error: {exc}"
        ) from exc

    if arr.ndim == 2:
        mode = "L"
    else:
        mode = "RGB"

    pil = Image.fromarray(arr, mode=mode)
    buf = BytesIO()
    pil.save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    decoded = Image.open(buf).convert(mode)
    out = np.asarray(decoded, dtype=np.uint8)
    return out, mask


def _blur_sigma_for_severity(severity: int) -> float:
    s = _validate_severity(severity)
    mapping = {
        1: 0.5,
        2: 1.0,
        3: 1.5,
        4: 2.0,
        5: 3.0,
    }
    return float(mapping[s])


def apply_blur(
    image: NDArray,
    *,
    mask: Optional[NDArray],
    severity: int,
    rng: np.random.Generator,
) -> tuple[NDArray, Optional[NDArray]]:
    """Apply a deterministic blur corruption (Gaussian blur)."""

    _ = rng  # reserved for future randomized variants; keep signature consistent
    arr = _as_u8_image(image)
    sigma = _blur_sigma_for_severity(severity)

    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "opencv-python is required for blur corruption.\n"
            "Install it via:\n  pip install 'opencv-python'\n"
            f"Original error: {exc}"
        ) from exc

    out = cv2.GaussianBlur(arr, ksize=(0, 0), sigmaX=float(sigma))
    out_u8 = np.asarray(out, dtype=np.uint8)
    return out_u8, mask


def apply_glare(
    image: NDArray,
    *,
    mask: Optional[NDArray],
    severity: int,
    rng: np.random.Generator,
) -> tuple[NDArray, Optional[NDArray]]:
    """Apply a deterministic glare/specular-highlight corruption."""

    arr = _as_u8_image(image)
    s = _validate_severity(severity)
    strength = float(s) / 5.0

    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "opencv-python is required for glare corruption.\n"
            "Install it via:\n  pip install 'opencv-python'\n"
            f"Original error: {exc}"
        ) from exc

    h, w = int(arr.shape[0]), int(arr.shape[1])
    if h == 0 or w == 0:
        return arr, mask

    glare_mask = np.zeros((h, w), dtype=np.float32)
    num_blobs = 1 + (s // 2)
    min_dim = float(min(h, w))
    radius_base = max(1.0, min_dim * (0.04 + 0.08 * strength))

    for _ in range(int(num_blobs)):
        cx = int(rng.integers(0, w))
        cy = int(rng.integers(0, h))
        r = int(max(1, round(radius_base * float(rng.uniform(0.8, 1.2)))))
        cv2.circle(glare_mask, (cx, cy), r, color=1.0, thickness=-1)

    sigma = float(0.5 + 2.0 * strength)
    glare_mask = cv2.GaussianBlur(glare_mask, ksize=(0, 0), sigmaX=sigma)
    glare_mask = np.clip(glare_mask, 0.0, 1.0)

    alpha = float(rng.uniform(0.3, 0.9) * strength)
    m = glare_mask * alpha

    out = arr.astype(np.float32)
    if out.ndim == 3:
        out = out + (255.0 - out) * m[..., None]
    else:
        out = out + (255.0 - out) * m

    out_u8 = np.clip(out, 0.0, 255.0).astype(np.uint8)
    return out_u8, mask


def apply_geo_jitter(
    image: NDArray,
    *,
    mask: Optional[NDArray],
    severity: int,
    rng: np.random.Generator,
) -> tuple[NDArray, Optional[NDArray]]:
    """Apply a deterministic geometric jitter (affine warp).

    Notes
    -----
    - The mask (if provided) is warped with nearest-neighbor interpolation.
    - The image is warped with bilinear interpolation.
    """

    arr = _as_u8_image(image)
    s = _validate_severity(severity)
    strength = float(s) / 5.0

    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "opencv-python is required for geo jitter corruption.\n"
            "Install it via:\n  pip install 'opencv-python'\n"
            f"Original error: {exc}"
        ) from exc

    h, w = int(arr.shape[0]), int(arr.shape[1])
    if h == 0 or w == 0:
        return arr, mask

    max_angle = 5.0 * strength
    max_shift = 0.05 * strength
    max_scale = 0.10 * strength

    angle = float(rng.uniform(-max_angle, max_angle))
    scale = float(rng.uniform(1.0 - max_scale, 1.0 + max_scale))
    dx = float(rng.uniform(-max_shift, max_shift) * w)
    dy = float(rng.uniform(-max_shift, max_shift) * h)

    center = ((w - 1) / 2.0, (h - 1) / 2.0)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    M[0, 2] += dx
    M[1, 2] += dy

    warped = cv2.warpAffine(
        arr,
        M,
        dsize=(w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    out_img = np.asarray(warped, dtype=np.uint8)

    out_mask = None
    if mask is not None:
        mask_arr = np.asarray(mask)
        if mask_arr.shape != (h, w):
            raise ValueError(
                "mask shape must match image spatial size. "
                f"Got mask={mask_arr.shape} vs image={(h, w)}."
            )
        mask_u8 = (mask_arr > 0).astype(np.uint8, copy=False)
        warped_mask = cv2.warpAffine(
            mask_u8,
            M,
            dsize=(w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        out_mask = (np.asarray(warped_mask, dtype=np.uint8) > 0).astype(np.uint8, copy=False)

    return out_img, out_mask
