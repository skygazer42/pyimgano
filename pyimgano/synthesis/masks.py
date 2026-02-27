from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


def _as_hw(shape_hw: tuple[int, int]) -> tuple[int, int]:
    h, w = int(shape_hw[0]), int(shape_hw[1])
    if h <= 0 or w <= 0:
        raise ValueError(f"shape_hw must be positive, got {shape_hw!r}")
    return h, w


def ensure_u8_mask(mask: np.ndarray, *, shape_hw: Optional[tuple[int, int]] = None) -> np.ndarray:
    """Normalize a mask to uint8 {0,255}."""

    arr = np.asarray(mask)
    if shape_hw is not None:
        h, w = _as_hw(shape_hw)
        if arr.shape != (h, w):
            raise ValueError(f"mask shape must be {(h, w)}, got {arr.shape}")

    if arr.dtype == np.bool_:
        out = arr.astype(np.uint8) * 255
        return out

    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        if float(np.max(arr)) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)

    return ((arr > 0).astype(np.uint8) * 255).astype(np.uint8)


def apply_roi_mask(mask_u8: np.ndarray, roi_mask: np.ndarray | None) -> np.ndarray:
    """Apply a ROI constraint to a binary mask (keeps anomalies inside ROI)."""

    mask = ensure_u8_mask(mask_u8)
    if roi_mask is None:
        return mask

    roi = ensure_u8_mask(roi_mask, shape_hw=mask.shape)
    return ((mask > 0) & (roi > 0)).astype(np.uint8) * 255


def random_blob_mask(
    shape_hw: tuple[int, int],
    *,
    rng: np.random.Generator,
    num_blobs: int = 3,
    radius_range: tuple[int, int] = (8, 48),
    blur_sigma: float = 0.0,
    threshold: float = 0.5,
) -> np.ndarray:
    """Random blob mask: filled circles + optional Gaussian blur + threshold."""

    import cv2  # local import to keep module import light

    h, w = _as_hw(shape_hw)
    n = max(1, int(num_blobs))
    r0, r1 = int(radius_range[0]), int(radius_range[1])
    if r0 < 1 or r1 < 1 or r0 > r1:
        raise ValueError(f"Invalid radius_range: {radius_range!r}")

    mask = np.zeros((h, w), dtype=np.uint8)
    for _ in range(n):
        cx = int(rng.integers(0, w))
        cy = int(rng.integers(0, h))
        r = int(rng.integers(r0, r1 + 1))
        cv2.circle(mask, (cx, cy), r, color=255, thickness=-1)

    if float(blur_sigma) > 0.0:
        m = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, ksize=(0, 0), sigmaX=float(blur_sigma))
        thr = float(np.clip(threshold, 0.0, 1.0))
        mask = ((m >= thr).astype(np.uint8) * 255).astype(np.uint8)

    return mask


def random_ellipse_mask(
    shape_hw: tuple[int, int],
    *,
    rng: np.random.Generator,
    num_ellipses: int = 2,
    axis_range: tuple[int, int] = (8, 64),
    blur_sigma: float = 0.0,
    threshold: float = 0.5,
) -> np.ndarray:
    """Random ellipse mask for stains/pits."""

    import cv2

    h, w = _as_hw(shape_hw)
    n = max(1, int(num_ellipses))
    a0, a1 = int(axis_range[0]), int(axis_range[1])
    if a0 < 1 or a1 < 1 or a0 > a1:
        raise ValueError(f"Invalid axis_range: {axis_range!r}")

    mask = np.zeros((h, w), dtype=np.uint8)
    for _ in range(n):
        cx = int(rng.integers(0, w))
        cy = int(rng.integers(0, h))
        ax = int(rng.integers(a0, a1 + 1))
        ay = int(rng.integers(a0, a1 + 1))
        angle = float(rng.uniform(0.0, 180.0))
        cv2.ellipse(mask, (cx, cy), (ax, ay), angle, 0.0, 360.0, color=255, thickness=-1)

    if float(blur_sigma) > 0.0:
        m = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, ksize=(0, 0), sigmaX=float(blur_sigma))
        thr = float(np.clip(threshold, 0.0, 1.0))
        mask = ((m >= thr).astype(np.uint8) * 255).astype(np.uint8)

    return mask


@dataclass(frozen=True)
class ScratchSpec:
    thickness: int = 2
    length_fraction: float = 0.4
    jitter: float = 0.15
    blur_sigma: float = 0.0


def random_scratch_mask(
    shape_hw: tuple[int, int],
    *,
    rng: np.random.Generator,
    num_scratches: int = 2,
    spec: ScratchSpec | None = None,
) -> np.ndarray:
    """Random scratch mask: thin polylines with optional blur."""

    import cv2

    h, w = _as_hw(shape_hw)
    n = max(1, int(num_scratches))
    s = ScratchSpec() if spec is None else spec

    mask = np.zeros((h, w), dtype=np.uint8)
    min_dim = float(min(h, w))
    length = max(4.0, float(s.length_fraction) * min_dim)

    for _ in range(n):
        x0 = float(rng.uniform(0.0, float(w - 1)))
        y0 = float(rng.uniform(0.0, float(h - 1)))
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        dx = math.cos(angle)
        dy = math.sin(angle)

        # Build a polyline with small jitter.
        steps = max(2, int(round(length / 12.0)))
        pts: list[tuple[int, int]] = []
        for i in range(steps + 1):
            t = float(i) / float(steps)
            jx = float(rng.normal(0.0, 1.0) * float(s.jitter) * 6.0)
            jy = float(rng.normal(0.0, 1.0) * float(s.jitter) * 6.0)
            x = x0 + (t - 0.5) * length * dx + jx
            y = y0 + (t - 0.5) * length * dy + jy
            xi = int(np.clip(round(x), 0, w - 1))
            yi = int(np.clip(round(y), 0, h - 1))
            pts.append((xi, yi))

        cv2.polylines(
            mask,
            [np.asarray(pts, dtype=np.int32).reshape(-1, 1, 2)],
            isClosed=False,
            color=255,
            thickness=max(1, int(s.thickness)),
            lineType=cv2.LINE_AA,
        )

    # Anti-aliased drawing produces intermediate values; normalize to binary.
    mask = ((mask > 0).astype(np.uint8) * 255).astype(np.uint8)

    if float(s.blur_sigma) > 0.0:
        blurred = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, ksize=(0, 0), sigmaX=float(s.blur_sigma))
        mask = ((blurred >= 0.2).astype(np.uint8) * 255).astype(np.uint8)

    return mask
