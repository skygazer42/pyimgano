"""Gabor filter bank feature extractor."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

from .base import BaseFeatureExtractor
from .registry import register_feature_extractor


def _to_gray_f32(img: np.ndarray) -> np.ndarray:
    import cv2

    arr = np.asarray(img)
    if arr.ndim == 2:
        gray = arr
    elif arr.ndim == 3 and arr.shape[2] == 3:
        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Unsupported image shape: {arr.shape}")
    return gray.astype(np.float32) / 255.0


def _load_image(item: Any) -> np.ndarray:
    from pathlib import Path

    if isinstance(item, (str, Path)):
        from pyimgano.io.image import read_image

        return np.asarray(read_image(item, color="bgr"))
    return np.asarray(item)


@register_feature_extractor(
    "gabor_bank",
    tags=("image", "texture"),
    metadata={"description": "Gabor filter bank response statistics (mean/std)"},
)
class GaborBankExtractor(BaseFeatureExtractor):
    def __init__(
        self,
        *,
        resize_hw: tuple[int, int] | None = (128, 128),
        frequencies: Sequence[float] = (0.1, 0.2, 0.3),
        thetas: Sequence[float] = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
        eps: float = 1e-12,
    ) -> None:
        self.resize_hw = resize_hw
        self.frequencies = tuple(float(f) for f in frequencies)
        self.thetas = tuple(float(t) for t in thetas)
        self.eps = float(eps)

    def extract(self, inputs: Iterable[Any]) -> np.ndarray:
        from pyimgano.utils.optional_deps import require
        import cv2

        skfilters = require("skimage.filters", extra="skimage", purpose="gabor_bank feature extractor")
        gabor = skfilters.gabor

        items = list(inputs)
        if not items:
            return np.zeros((0, 0), dtype=np.float32)

        feats: list[np.ndarray] = []
        for item in items:
            img = _load_image(item)
            if self.resize_hw is not None:
                h, w = int(self.resize_hw[0]), int(self.resize_hw[1])
                img = cv2.resize(np.asarray(img), (w, h), interpolation=cv2.INTER_AREA)

            gray = _to_gray_f32(img)

            fvec: list[float] = []
            for freq in self.frequencies:
                for theta in self.thetas:
                    real, imag = gabor(gray, frequency=float(freq), theta=float(theta))
                    mag = np.sqrt(real * real + imag * imag)
                    fvec.append(float(np.mean(mag)))
                    fvec.append(float(np.std(mag)))
            feats.append(np.asarray(fvec, dtype=np.float32))

        return np.stack(feats, axis=0)
