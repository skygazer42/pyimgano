# -*- coding: utf-8 -*-
"""Legacy structural Isolation Forest model (modernized).

`isolation_forest_struct` previously existed as a standalone, UI/screen-focused
structural baseline.

This module keeps the registry name stable but rebuilds the implementation on:
- `StructuralFeaturesExtractor` (handcrafted edges/shape/texture)
- `vision_iforest` (sklearn IsolationForest backend)

The goal is consistency: BaseDetector semantics, feature registry support,
feature caching support, and predictable CLI behavior.
"""

from __future__ import annotations

from typing import Optional, Union

from pyimgano.features.structural import StructuralFeaturesExtractor

from .iforest import VisionIForest
from .registry import register_model


@register_model(
    "isolation_forest_struct",
    tags=("vision", "classical", "iforest", "ensemble", "structural"),
    metadata={
        "description": "Structural-features Isolation Forest baseline (modernized; native base classes)",
        "legacy_name": True,
    },
    overwrite=True,
)
class IsolationForestStruct(VisionIForest):
    """Backwards-compatible alias for IsolationForest on structural features."""

    def __init__(
        self,
        *,
        feature_extractor=None,
        contamination: float = 0.01,
        n_estimators: int = 100,
        max_samples: Union[str, int] = "auto",
        max_features: float = 1.0,
        bootstrap: bool = False,
        n_jobs: int = 1,
        random_state: Optional[int] = None,
        verbose: int = 0,
        **kwargs,
    ) -> None:
        if feature_extractor is None:
            feature_extractor = StructuralFeaturesExtractor(max_size=512, error_mode="zeros")

        super().__init__(
            feature_extractor=feature_extractor,
            contamination=float(contamination),
            n_estimators=int(n_estimators),
            max_samples=max_samples,
            max_features=float(max_features),
            bootstrap=bool(bootstrap),
            n_jobs=int(n_jobs),
            random_state=random_state,
            verbose=int(verbose),
            **dict(kwargs),
        )
