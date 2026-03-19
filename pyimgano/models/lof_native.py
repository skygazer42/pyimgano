# -*- coding: utf-8 -*-
"""Vision-friendly Local Outlier Factor (LOF) wrapper."""

from __future__ import annotations

from typing import Optional

from ._legacy_x import MISSING, resolve_legacy_x_keyword
from .baseml import BaseVisionDetector
from .lof_core import CoreLOF
from .registry import register_model


@register_model(
    "vision_lof",
    tags=("vision", "classical", "lof", "neighbors", "density"),
    metadata={
        "description": "Vision wrapper for LOF (Local Outlier Factor, novelty mode)",
        "paper": "Breunig et al., SIGMOD 2000",
        "year": 2000,
    },
)
class VisionLOF(BaseVisionDetector):
    """Vision-compatible LOF detector for anomaly detection."""

    def __init__(
        self,
        *,
        feature_extractor=None,
        contamination: float = 0.1,
        n_neighbors: int = 20,
        metric: str = "minkowski",
        p: int = 2,
        leaf_size: int = 30,
        n_jobs: Optional[int] = None,
    ) -> None:
        self._detector_kwargs = {
            "contamination": float(contamination),
            "n_neighbors": int(n_neighbors),
            "metric": str(metric),
            "p": int(p),
            "leaf_size": int(leaf_size),
            "n_jobs": n_jobs,
        }
        super().__init__(contamination=contamination, feature_extractor=feature_extractor)

    def _build_detector(self):
        return CoreLOF(**self._detector_kwargs)

    def fit(self, x: object = MISSING, y=None, **kwargs: object):
        return super().fit(resolve_legacy_x_keyword(x, kwargs, method_name="fit"), y=y)

    def decision_function(self, x: object = MISSING, **kwargs: object):
        return super().decision_function(resolve_legacy_x_keyword(x, kwargs, method_name="decision_function"))
