# -*- coding: utf-8 -*-
"""SO-GAAL (Single-Objective Generative Adversarial Active Learning) wrapper.

SO-GAAL is implemented in PyOD. In PyImgAno, we apply it to image feature
vectors produced by a pluggable feature extractor.
"""

from __future__ import annotations

from pyod.models.so_gaal import SO_GAAL

from .baseml import BaseVisionDetector
from .registry import register_model


@register_model(
    "vision_so_gaal",
    tags=("vision", "deep", "gan", "so_gaal", "pyod"),
    metadata={"description": "SO-GAAL wrapper via PyOD"},
)
class VisionSOGaal(BaseVisionDetector):
    """Vision-friendly wrapper around PyOD's SO-GAAL detector."""

    def __init__(
        self,
        *,
        feature_extractor=None,
        contamination: float = 0.1,
        stop_epochs: int = 20,
        lr_d: float = 0.01,
        lr_g: float = 0.0001,
        momentum: float = 0.9,
    ) -> None:
        self._detector_kwargs = {
            "stop_epochs": int(stop_epochs),
            "lr_d": float(lr_d),
            "lr_g": float(lr_g),
            "momentum": float(momentum),
        }
        super().__init__(contamination=contamination, feature_extractor=feature_extractor)

    def _build_detector(self):
        return SO_GAAL(contamination=self.contamination, **self._detector_kwargs)

