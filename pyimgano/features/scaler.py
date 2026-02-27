"""StandardScaler feature extractor (fit + transform)."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from sklearn.preprocessing import StandardScaler

from .base import BaseFeatureExtractor
from .identity import IdentityExtractor
from .protocols import FeatureExtractor, FittableFeatureExtractor
from .registry import register_feature_extractor, resolve_feature_extractor


@register_feature_extractor(
    "standard_scaler",
    tags=("pipeline", "scaling"),
    metadata={"description": "Apply StandardScaler on top of a base extractor"},
)
class StandardScalerExtractor(BaseFeatureExtractor):
    def __init__(
        self,
        *,
        base_extractor: FeatureExtractor | str | dict | None = None,
        with_mean: bool = True,
        with_std: bool = True,
    ) -> None:
        if base_extractor is None:
            base_extractor = IdentityExtractor()
        self.base_extractor = resolve_feature_extractor(base_extractor)
        self.with_mean = bool(with_mean)
        self.with_std = bool(with_std)
        self._scaler: StandardScaler | None = None

    def fit(self, inputs: Iterable[Any], y: Any | None = None) -> "StandardScalerExtractor":
        items = list(inputs)
        base = self.base_extractor
        if isinstance(base, FittableFeatureExtractor):
            base.fit(items, y=y)

        X = np.asarray(base.extract(items), dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        scaler = StandardScaler(with_mean=self.with_mean, with_std=self.with_std)
        scaler.fit(X)
        self._scaler = scaler
        return self

    def extract(self, inputs: Iterable[Any]) -> np.ndarray:
        if self._scaler is None:
            raise RuntimeError("StandardScalerExtractor is not fitted yet. Call fit() first.")

        items = list(inputs)
        X = np.asarray(self.base_extractor.extract(items), dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        Z = self._scaler.transform(X)
        return np.asarray(Z, dtype=np.float32)
