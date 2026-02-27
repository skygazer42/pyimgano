from __future__ import annotations

from typing import Literal

import numpy as np


def _as_u8_image(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        raise ValueError(f"Expected uint8 image, got dtype={arr.dtype}")
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[2] == 3:
        return arr
    raise ValueError(f"Expected grayscale (H,W) or color (H,W,3) image, got {arr.shape}")


def _validate_severity(severity: int) -> int:
    s = int(severity)
    if s < 1 or s > 5:
        raise ValueError(f"severity must be in [1,5], got {severity}")
    return s


def vibration_blur(
    image_u8: np.ndarray,
    *,
    severity: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate vibration/motion blur common in fast conveyor imaging."""

    import cv2  # local import

    img = _as_u8_image(image_u8)
    s = _validate_severity(severity)
    k = int({1: 3, 2: 5, 3: 7, 4: 11, 5: 15}[s])
    k = max(3, k | 1)  # odd

    angle = float(rng.uniform(0.0, np.pi))
    dx = float(np.cos(angle))
    dy = float(np.sin(angle))

    kernel = np.zeros((k, k), dtype=np.float32)
    cx = (k - 1) / 2.0
    cy = (k - 1) / 2.0
    for i in range(k):
        x = cx + (i - cx) * dx
        y = cy + (i - cy) * dy
        xi = int(np.clip(round(x), 0, k - 1))
        yi = int(np.clip(round(y), 0, k - 1))
        kernel[yi, xi] += 1.0
    kernel = kernel / float(np.sum(kernel))

    out = cv2.filter2D(img, ddepth=-1, kernel=kernel, borderType=cv2.BORDER_REFLECT_101)
    return np.asarray(out, dtype=np.uint8)


def stripe_noise(
    image_u8: np.ndarray,
    *,
    severity: int,
    rng: np.random.Generator,
    direction: Literal["horizontal", "vertical"] = "horizontal",
) -> np.ndarray:
    """Add banding/stripe noise (sensor/lighting artifacts)."""

    img = _as_u8_image(image_u8)
    s = _validate_severity(severity)
    strength = float(s) / 5.0

    h, w = int(img.shape[0]), int(img.shape[1])
    if h == 0 or w == 0:
        return np.asarray(img, dtype=np.uint8)

    # Random low-frequency sinusoidal pattern + random offsets.
    freq = float(rng.uniform(1.0, 4.0))
    amp = float(rng.uniform(6.0, 28.0) * strength)

    if direction == "horizontal":
        yy = np.linspace(0.0, 2.0 * np.pi * freq, h, dtype=np.float32)
        pat = (np.sin(yy) * amp).astype(np.float32).reshape(h, 1)
    else:
        xx = np.linspace(0.0, 2.0 * np.pi * freq, w, dtype=np.float32)
        pat = (np.sin(xx) * amp).astype(np.float32).reshape(1, w)

    jitter = rng.normal(0.0, 1.0, size=(h, 1) if direction == "horizontal" else (1, w)).astype(
        np.float32
    )
    pat = pat + jitter * (amp * 0.15)

    out = img.astype(np.float32)
    if out.ndim == 3:
        out = out + pat[..., None]
    else:
        out = out + pat
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def dust_specks(
    image_u8: np.ndarray,
    *,
    severity: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add small dust specks / debris artifacts."""

    import cv2  # local import

    img = _as_u8_image(image_u8)
    s = _validate_severity(severity)
    strength = float(s) / 5.0

    h, w = int(img.shape[0]), int(img.shape[1])
    if h == 0 or w == 0:
        return np.asarray(img, dtype=np.uint8)

    num = int(round((h * w) / 8000.0 * (1.0 + 3.0 * strength)))
    num = max(1, min(num, 500))

    out = np.array(img, copy=True)
    for _ in range(num):
        x = int(rng.integers(0, w))
        y = int(rng.integers(0, h))
        r = int(rng.integers(1, 3 + int(3 * strength)))
        val = int(rng.integers(180, 255))
        if out.ndim == 3:
            cv2.circle(out, (x, y), r, color=(val, val, val), thickness=-1, lineType=cv2.LINE_AA)
        else:
            cv2.circle(out, (x, y), r, color=val, thickness=-1, lineType=cv2.LINE_AA)

    return np.asarray(out, dtype=np.uint8)

