"""Histogram of Oriented Gradients (HOG) feature extractor."""

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
        # Treat as BGR/RGB agnostic: conversion only uses channel values.
        return cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY).astype(np.uint8, copy=False)
    raise ValueError(f"Unsupported image shape: {arr.shape}")


def _load_image(item: Any) -> np.ndarray:
    from pathlib import Path

    if isinstance(item, (str, Path)):
        from pyimgano.io.image import read_image

        return np.asarray(read_image(item, color="bgr"))
    return np.asarray(item)


@register_feature_extractor(
    "hog",
    tags=("image", "texture"),
    metadata={"description": "HOG (Histogram of Oriented Gradients) features"},
)
class HOGExtractor(BaseFeatureExtractor):
    def __init__(
        self,
        *,
        resize_hw: tuple[int, int] | None = (128, 128),
        orientations: int = 9,
        pixels_per_cell: tuple[int, int] = (8, 8),
        cells_per_block: tuple[int, int] = (2, 2),
        block_norm: str = "L2-Hys",
        transform_sqrt: bool = True,
    ) -> None:
        self.resize_hw = resize_hw
        self.orientations = int(orientations)
        self.pixels_per_cell = tuple(int(x) for x in pixels_per_cell)
        self.cells_per_block = tuple(int(x) for x in cells_per_block)
        self.block_norm = str(block_norm)
        self.transform_sqrt = bool(transform_sqrt)

    def extract(self, inputs: Iterable[Any]) -> np.ndarray:
        from skimage.feature import hog
        import cv2

        items = list(inputs)
        if not items:
            return np.zeros((0, 0), dtype=np.float32)

        feats: list[np.ndarray] = []
        for item in items:
            img = _load_image(item)
            gray = _to_gray_u8(img)
            if self.resize_hw is not None:
                h, w = int(self.resize_hw[0]), int(self.resize_hw[1])
                gray = cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA)

            f = hog(
                gray,
                orientations=self.orientations,
                pixels_per_cell=self.pixels_per_cell,
                cells_per_block=self.cells_per_block,
                block_norm=self.block_norm,
                transform_sqrt=self.transform_sqrt,
                feature_vector=True,
            )
            feats.append(np.asarray(f, dtype=np.float32).reshape(-1))

        return np.stack(feats, axis=0)

