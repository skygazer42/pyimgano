"""Identity feature extractor.

This is useful for "feature-based" detectors when inputs are already vectors
(tabular data, embeddings, precomputed descriptors, ...).
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from .base import BaseFeatureExtractor
from .registry import register_feature_extractor


@register_feature_extractor(
    "identity",
    tags=("tabular", "embeddings", "noop"),
    metadata={"description": "Identity extractor (inputs are already feature vectors)."},
)
class IdentityExtractor(BaseFeatureExtractor):
    def extract(self, inputs: Iterable[Any]) -> np.ndarray:
        return np.asarray(list(inputs))
