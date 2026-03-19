"""PCA projection feature extractor (fit + transform)."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from sklearn.decomposition import PCA

from .base import BaseFeatureExtractor
from .identity import IdentityExtractor
from .protocols import FeatureExtractor, FittableFeatureExtractor
from .registry import register_feature_extractor, resolve_feature_extractor


@register_feature_extractor(
    "pca_projector",
    tags=("pipeline", "pca"),
    metadata={"description": "Apply PCA projection on top of a base extractor"},
)
class PCAProjector(BaseFeatureExtractor):
    def __init__(
        self,
        *,
        base_extractor: FeatureExtractor | str | dict | None = None,
        n_components: int | float = 0.95,
        whiten: bool = False,
        random_state: int | None = None,
    ) -> None:
        if base_extractor is None:
            base_extractor = IdentityExtractor()
        # Allow JSON-friendly specs: {"name": "...", "kwargs": {...}}.
        self.base_extractor = resolve_feature_extractor(base_extractor)
        self.n_components = n_components
        self.whiten = bool(whiten)
        self.random_state = random_state
        self._pca: PCA | None = None

    def fit(self, inputs: Iterable[Any], y: Any | None = None) -> "PCAProjector":
        items = list(inputs)
        base = self.base_extractor
        if isinstance(base, FittableFeatureExtractor):
            base.fit(items, y=y)

        x = np.asarray(base.extract(items), dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        pca = PCA(
            n_components=self.n_components, whiten=self.whiten, random_state=self.random_state
        )
        pca.fit(x)
        self._pca = pca
        return self

    def extract(self, inputs: Iterable[Any]) -> np.ndarray:
        if self._pca is None:
            raise RuntimeError("PCAProjector is not fitted yet. Call fit() first.")

        items = list(inputs)
        x = np.asarray(self.base_extractor.extract(items), dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        z = self._pca.transform(x)
        return np.asarray(z, dtype=np.float32)
