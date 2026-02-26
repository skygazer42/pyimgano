"""Edge statistics feature extractor (Canny + Sobel)."""

from __future__ import annotations

from typing import Any, Iterable

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
    "edge_stats",
    tags=("image", "edges"),
    metadata={"description": "Edge statistics from Canny and Sobel gradients"},
)
class EdgeStatsExtractor(BaseFeatureExtractor):
    def __init__(
        self,
        *,
        canny_threshold1: int = 50,
        canny_threshold2: int = 150,
        sobel_ksize: int = 3,
    ) -> None:
        self.canny_threshold1 = int(canny_threshold1)
        self.canny_threshold2 = int(canny_threshold2)
        self.sobel_ksize = int(sobel_ksize)

    def extract(self, inputs: Iterable[Any]) -> np.ndarray:
        import cv2

        items = list(inputs)
        if not items:
            return np.zeros((0, 0), dtype=np.float32)

        feats: list[np.ndarray] = []
        for item in items:
            img = _load_image(item)
            gray = _to_gray_u8(img)

            edges = cv2.Canny(gray, self.canny_threshold1, self.canny_threshold2)
            edges_f = edges.astype(np.float32) / 255.0

            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=self.sobel_ksize)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=self.sobel_ksize)
            mag = np.sqrt(gx * gx + gy * gy) / 255.0

            vec = np.asarray(
                [
                    float(np.mean(edges_f)),
                    float(np.std(edges_f)),
                    float(np.mean(edges_f > 0.0)),
                    float(np.mean(mag)),
                    float(np.std(mag)),
                    float(np.percentile(mag, 90)),
                ],
                dtype=np.float32,
            )
            feats.append(vec)

        return np.stack(feats, axis=0)

