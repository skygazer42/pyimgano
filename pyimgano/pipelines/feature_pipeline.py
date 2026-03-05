"""Generic feature-extractor + core-detector pipeline.

Use this when you want to combine a registered core detector with a registered
feature extractor without writing a dedicated `vision_*` wrapper class.
"""

from __future__ import annotations

from typing import Any, Mapping

from pyimgano.models.baseml import BaseVisionDetector
from pyimgano.models.registry import create_model


class VisionFeaturePipeline(BaseVisionDetector):
    def __init__(
        self,
        *,
        core_detector: str | type | Any,
        core_kwargs: Mapping[str, Any] | None = None,
        feature_extractor=None,
        contamination: float = 0.1,
    ) -> None:
        self.core_detector = core_detector
        self.core_kwargs = dict(core_kwargs or {})
        super().__init__(contamination=contamination, feature_extractor=feature_extractor)

    def _build_detector(self):
        cd = self.core_detector
        if isinstance(cd, str):
            return create_model(cd, contamination=self.contamination, **self.core_kwargs)
        if isinstance(cd, type):
            return cd(contamination=self.contamination, **self.core_kwargs)
        return cd
