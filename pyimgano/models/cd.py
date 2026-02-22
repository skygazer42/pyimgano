# -*- coding: utf-8 -*-
"""Cook's Distance (CD) vision wrapper.

CD is a regression-influence-based outlier detector implemented in PyOD.
In PyImgAno, we apply it to image feature vectors produced by a pluggable
feature extractor.
"""

from __future__ import annotations

from typing import Any, Optional

from pyod.models.cd import CD

from .baseml import BaseVisionDetector
from .registry import register_model


@register_model(
    "vision_cd",
    tags=("vision", "classical", "cd", "pyod"),
    metadata={"description": "Cook's Distance (CD) wrapper via PyOD"},
)
class VisionCD(BaseVisionDetector):
    """Vision-friendly wrapper around PyOD's Cook's Distance outlier detector."""

    def __init__(
        self,
        *,
        feature_extractor=None,
        contamination: float = 0.1,
        model: Optional[Any] = None,
    ) -> None:
        self.model = model
        super().__init__(contamination=contamination, feature_extractor=feature_extractor)

    def _build_detector(self):
        return CD(contamination=self.contamination, model=self.model)

