"""Color histogram feature extractor."""

from __future__ import annotations

from typing import Any, Iterable, Literal

import numpy as np

from .base import BaseFeatureExtractor
from .registry import register_feature_extractor

_ColorSpace = Literal["bgr", "rgb", "hsv", "lab"]


def _load_image(item: Any) -> np.ndarray:
    from pathlib import Path

    if isinstance(item, (str, Path)):
        from pyimgano.io.image import read_image

        return np.asarray(read_image(item, color="bgr"))
    return np.asarray(item)


@register_feature_extractor(
    "color_hist",
    tags=("image", "color"),
    metadata={"description": "Per-channel color histogram (BGR/RGB/HSV/LAB)"},
)
class ColorHistogramExtractor(BaseFeatureExtractor):
    def __init__(
        self,
        *,
        colorspace: _ColorSpace = "hsv",
        bins: tuple[int, int, int] = (16, 16, 16),
        eps: float = 1e-12,
    ) -> None:
        self.colorspace = str(colorspace).lower()
        self.bins = tuple(int(b) for b in bins)
        self.eps = float(eps)

    def extract(self, inputs: Iterable[Any]) -> np.ndarray:
        import cv2

        items = list(inputs)
        if not items:
            return np.zeros((0, 0), dtype=np.float32)

        feats: list[np.ndarray] = []
        for item in items:
            img = _load_image(item)
            if img.ndim != 3 or img.shape[2] != 3:
                raise ValueError(f"Expected image shape (H,W,3), got {img.shape}")

            cs = self.colorspace
            if cs == "bgr":
                x = img
            elif cs == "rgb":
                x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif cs == "hsv":
                x = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            elif cs == "lab":
                x = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            else:
                raise ValueError("colorspace must be one of: bgr, rgb, hsv, lab")

            hists: list[np.ndarray] = []
            for c in range(3):
                ch = x[:, :, c]
                b = int(self.bins[c])
                if b <= 0:
                    raise ValueError("bins must be positive integers")

                if cs == "hsv" and c == 0:
                    rng = (0, 180)
                else:
                    rng = (0, 256)
                hist, _ = np.histogram(ch.ravel(), bins=b, range=rng)
                hists.append(hist.astype(np.float32))

            vec = np.concatenate(hists, axis=0)
            vec = vec / float(np.sum(vec) + float(self.eps))
            feats.append(vec)

        return np.stack(feats, axis=0)

