"""Feature normalization extractor (power + L2 normalization)."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from .base import BaseFeatureExtractor
from .identity import IdentityExtractor
from .protocols import FeatureExtractor, FittableFeatureExtractor
from .registry import register_feature_extractor, resolve_feature_extractor


@register_feature_extractor(
    "normalize",
    tags=("pipeline", "normalize"),
    metadata={"description": "Apply power and/or L2 normalization on top of a base extractor"},
)
class NormalizeExtractor(BaseFeatureExtractor):
    """Normalize feature rows from a base extractor.

    This is commonly used for embedding pipelines:
    - power normalization (burstiness reduction): sign(x) * |x|^power
    - L2 normalization: x / ||x||
    """

    def __init__(
        self,
        *,
        base_extractor: FeatureExtractor | str | dict | None = None,
        l2: bool = True,
        power: float | None = None,
        eps: float = 1e-12,
    ) -> None:
        if base_extractor is None:
            base_extractor = IdentityExtractor()
        self.base_extractor = resolve_feature_extractor(base_extractor)
        self.l2 = bool(l2)
        self.power = power
        self.eps = float(eps)

    def fit(self, inputs: Iterable[Any], y: Any | None = None) -> "NormalizeExtractor":
        items = list(inputs)
        base = self.base_extractor
        if isinstance(base, FittableFeatureExtractor):
            base.fit(items, y=y)
        return self

    def extract(self, inputs: Iterable[Any]) -> np.ndarray:
        items = list(inputs)
        x = np.asarray(self.base_extractor.extract(items), dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        out = x
        p = self.power
        if p is not None:
            power_f = float(p)
            if power_f <= 0.0:
                raise ValueError("power must be > 0 when provided")
            # Common choice: power=0.5 (signed sqrt).
            out = np.sign(out) * (np.abs(out) ** power_f)

        if self.l2:
            eps = max(float(self.eps), 1e-18)
            norms = np.linalg.norm(out, axis=1, keepdims=True)
            norms = np.maximum(norms, eps)
            out = out / norms

        return np.asarray(out, dtype=np.float32)
