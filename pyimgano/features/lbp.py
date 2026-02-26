"""Local Binary Pattern (LBP) feature extractor."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from .base import BaseFeatureExtractor
from .registry import register_feature_extractor


def _to_gray_u8(img: np.ndarray) -> np.ndarray:
    import cv2

    arr = np.asarray(img)
    if arr.ndim == 2:
        return arr.astype(np.uint8, copy=False)
    if arr.ndim == 3 and arr.shape[2] == 3:
        return cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY).astype(np.uint8, copy=False)
    raise ValueError(f"Unsupported image shape: {arr.shape}")


def _load_image(item: Any) -> np.ndarray:
    from pathlib import Path

    if isinstance(item, (str, Path)):
        from pyimgano.io.image import read_image

        return np.asarray(read_image(item, color="bgr"))
    return np.asarray(item)


@register_feature_extractor(
    "lbp",
    tags=("image", "texture"),
    metadata={"description": "LBP (Local Binary Pattern) histogram features"},
)
class LBPExtractor(BaseFeatureExtractor):
    def __init__(
        self,
        *,
        n_points: int = 8,
        radius: float = 1.0,
        method: str = "uniform",
        n_bins: int | None = None,
        eps: float = 1e-12,
    ) -> None:
        self.n_points = int(n_points)
        self.radius = float(radius)
        self.method = str(method)
        self.n_bins = None if n_bins is None else int(n_bins)
        self.eps = float(eps)

    def _resolve_bins(self) -> int:
        if self.n_bins is not None:
            return int(self.n_bins)
        m = self.method.lower()
        if m == "uniform":
            return int(self.n_points + 2)
        if m in {"default", "ror"}:
            return int(2**self.n_points)
        # For other modes (e.g. "var"), fall back to a small fixed histogram.
        return 16

    def extract(self, inputs: Iterable[Any]) -> np.ndarray:
        from skimage.feature import local_binary_pattern

        items = list(inputs)
        if not items:
            return np.zeros((0, 0), dtype=np.float32)

        bins = int(self._resolve_bins())
        feats: list[np.ndarray] = []
        for item in items:
            img = _load_image(item)
            gray = _to_gray_u8(img)

            lbp = local_binary_pattern(gray, P=self.n_points, R=self.radius, method=self.method)
            if lbp.dtype.kind == "f":
                hist, _ = np.histogram(lbp.ravel(), bins=bins, range=(float(lbp.min()), float(lbp.max()) + 1e-6))
            else:
                hist, _ = np.histogram(lbp.ravel(), bins=bins, range=(0, bins))
            hist = hist.astype(np.float32)
            hist = hist / float(np.sum(hist) + float(self.eps))
            feats.append(hist.reshape(-1))

        return np.stack(feats, axis=0)

