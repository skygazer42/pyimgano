"""FFT low-frequency energy feature extractor."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

from .base import BaseFeatureExtractor
from .registry import register_feature_extractor


def _load_image(item: Any) -> np.ndarray:
    from pathlib import Path

    if isinstance(item, (str, Path)):
        from pyimgano.io.image import read_image

        return np.asarray(read_image(item, color="bgr"))
    return np.asarray(item)


def _to_gray_u8(img: np.ndarray) -> np.ndarray:
    import cv2

    arr = np.asarray(img)
    if arr.ndim == 2:
        return arr.astype(np.uint8, copy=False)
    if arr.ndim == 3 and arr.shape[2] == 3:
        return cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY).astype(np.uint8, copy=False)
    raise ValueError(f"Unsupported image shape: {arr.shape}")


@register_feature_extractor(
    "fft_lowfreq",
    tags=("image", "frequency"),
    metadata={"description": "Low-frequency energy ratios from FFT magnitude"},
)
class FFTLowFreqExtractor(BaseFeatureExtractor):
    def __init__(
        self,
        *,
        size_hw: tuple[int, int] = (64, 64),
        radii: Sequence[int] = (4, 8, 16),
        eps: float = 1e-12,
    ) -> None:
        self.size_hw = (int(size_hw[0]), int(size_hw[1]))
        self.radii = tuple(int(r) for r in radii)
        self.eps = float(eps)

    def extract(self, inputs: Iterable[Any]) -> np.ndarray:
        import cv2

        items = list(inputs)
        if not items:
            return np.zeros((0, 0), dtype=np.float32)

        feats: list[np.ndarray] = []
        h, w = int(self.size_hw[0]), int(self.size_hw[1])
        cy, cx = h // 2, w // 2

        for item in items:
            img = _load_image(item)
            gray = _to_gray_u8(img)
            gray = cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA)

            f = np.fft.fft2(gray.astype(np.float32))
            fshift = np.fft.fftshift(f)
            mag = np.abs(fshift)
            total = float(np.sum(mag) + float(self.eps))

            vec: list[float] = []
            for r in self.radii:
                rr = int(r)
                y0, y1 = max(0, cy - rr), min(h, cy + rr)
                x0, x1 = max(0, cx - rr), min(w, cx + rr)
                low = float(np.sum(mag[y0:y1, x0:x1]))
                vec.append(low / total)

            feats.append(np.asarray(vec, dtype=np.float32))

        return np.stack(feats, axis=0)

