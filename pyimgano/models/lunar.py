# -*- coding: utf-8 -*-
"""LUNAR vision wrapper.

LUNAR is implemented in PyOD. In PyImgAno, we apply it to image feature vectors
produced by a pluggable feature extractor.
"""

from __future__ import annotations

from typing import Any, Optional

from pyod.models.lunar import LUNAR

from .baseml import BaseVisionDetector
from .registry import register_model


@register_model(
    "vision_lunar",
    tags=("vision", "deep", "lunar", "pyod"),
    metadata={"description": "LUNAR wrapper via PyOD"},
)
class VisionLUNAR(BaseVisionDetector):
    """Vision-friendly wrapper around PyOD's LUNAR detector."""

    def __init__(
        self,
        *,
        feature_extractor=None,
        contamination: float = 0.1,
        model_type: str = "WEIGHT",
        n_neighbours: int = 5,
        negative_sampling: str = "MIXED",
        val_size: float = 0.1,
        scaler: Optional[Any] = None,
        epsilon: float = 0.1,
        proportion: float = 1.0,
        n_epochs: int = 200,
        lr: float = 1e-3,
        wd: float = 0.1,
        verbose: int = 0,
    ) -> None:
        self._detector_kwargs = {
            "model_type": str(model_type),
            "n_neighbours": int(n_neighbours),
            "negative_sampling": str(negative_sampling),
            "val_size": float(val_size),
            "epsilon": float(epsilon),
            "proportion": float(proportion),
            "n_epochs": int(n_epochs),
            "lr": float(lr),
            "wd": float(wd),
            "verbose": int(verbose),
        }
        if scaler is not None:
            self._detector_kwargs["scaler"] = scaler

        super().__init__(contamination=contamination, feature_extractor=feature_extractor)

    def _build_detector(self):
        return LUNAR(contamination=self.contamination, **self._detector_kwargs)

