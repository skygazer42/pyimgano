# -*- coding: utf-8 -*-
"""Legacy structural LOF model (modernized).

Historically, `lof_structure` shipped as a standalone script-like detector with
ad-hoc feature extraction.

This module modernizes it by rebuilding the model around the current
`BaseVisionDetector` + feature-extractor contract:
- structural features via `StructuralFeaturesExtractor`
- LOF scoring via sklearn's `LocalOutlierFactor` (novelty mode)
- consistent thresholding via `BaseDetector`

The registry name `lof_structure` is kept stable for backwards compatibility.
"""

from __future__ import annotations

from typing import Optional

from pyimgano.features.structural import StructuralFeaturesExtractor

from .lof_native import VisionLOF
from .registry import register_model


@register_model(
    "lof_structure",
    tags=("vision", "classical", "lof", "neighbors", "structural"),
    metadata={
        "description": "Structural-features LOF anomaly detector (modernized; native base classes)",
        "legacy_name": True,
    },
    overwrite=True,
)
class LOFStructure(VisionLOF):
    """Backwards-compatible alias for LOF on structural (edge/shape) features."""

    def __init__(
        self,
        *,
        feature_extractor=None,
        contamination: float = 0.01,
        n_neighbors: int = 50,
        leaf_size: int = 30,
        metric: str = "minkowski",
        p: int = 2,
        n_jobs: Optional[int] = None,
    ) -> None:
        if feature_extractor is None:
            feature_extractor = StructuralFeaturesExtractor(max_size=512, error_mode="zeros")

        super().__init__(
            feature_extractor=feature_extractor,
            contamination=float(contamination),
            n_neighbors=int(n_neighbors),
            metric=str(metric),
            p=int(p),
            leaf_size=int(leaf_size),
            n_jobs=n_jobs,
        )
