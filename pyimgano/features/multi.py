"""MultiExtractor: concatenate features from multiple extractors."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

from .base import BaseFeatureExtractor
from .protocols import FeatureExtractor, FittableFeatureExtractor
from .registry import register_feature_extractor


@register_feature_extractor(
    "multi",
    tags=("pipeline",),
    metadata={"description": "Concatenate outputs of multiple feature extractors"},
)
class MultiExtractor(BaseFeatureExtractor):
    def __init__(self, extractors: Sequence[FeatureExtractor]) -> None:
        if not extractors:
            raise ValueError("extractors must be non-empty")
        self.extractors = list(extractors)

    def fit(self, inputs: Iterable[Any], y: Any | None = None) -> "MultiExtractor":
        items = list(inputs)
        for ext in self.extractors:
            if isinstance(ext, FittableFeatureExtractor):
                ext.fit(items, y=y)
        return self

    def extract(self, inputs: Iterable[Any]) -> np.ndarray:
        items = list(inputs)
        if not items:
            return np.zeros((0, 0), dtype=np.float32)

        mats: list[np.ndarray] = []
        for ext in self.extractors:
            m = np.asarray(ext.extract(items))
            if m.ndim == 1:
                m = m.reshape(-1, 1)
            if m.shape[0] != len(items):
                raise ValueError("Each extractor must return one row per input")
            mats.append(m.astype(np.float32, copy=False))

        return np.concatenate(mats, axis=1)

