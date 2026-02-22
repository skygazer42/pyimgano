# -*- coding: utf-8 -*-
"""Sampling-based outlier detection vision wrapper.

Sampling is implemented in PyOD. In PyImgAno, we apply it to image feature
vectors produced by a pluggable feature extractor.
"""

from __future__ import annotations

from typing import Any, Optional

from pyod.models.sampling import Sampling

from .baseml import BaseVisionDetector
from .registry import register_model


@register_model(
    "vision_sampling",
    tags=("vision", "classical", "sampling", "pyod"),
    metadata={"description": "Sampling wrapper via PyOD"},
)
class VisionSampling(BaseVisionDetector):
    """Vision-friendly wrapper around PyOD's Sampling detector."""

    def __init__(
        self,
        *,
        feature_extractor=None,
        contamination: float = 0.1,
        subset_size: int = 20,
        metric: str = "minkowski",
        metric_params: Optional[dict[str, Any]] = None,
        random_state: Optional[int] = None,
    ) -> None:
        self._detector_kwargs = {
            "subset_size": int(subset_size),
            "metric": str(metric),
            "metric_params": metric_params,
            "random_state": random_state,
        }
        super().__init__(contamination=contamination, feature_extractor=feature_extractor)

    def _build_detector(self):
        return Sampling(contamination=self.contamination, **self._detector_kwargs)

