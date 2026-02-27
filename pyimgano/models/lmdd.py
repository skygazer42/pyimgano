# -*- coding: utf-8 -*-
"""
LMDD (Linear Model Deviation-based Detector).

LMDD employs the concept of a smoothing factor which indicates how much the
dissimilarity can be reduced by removing a subset of elements from the dataset.

Reference:
    Arning, A., Agrawal, R. and Raghavan, P., 1996.
    A Linear Method for Deviation Detection in Large Databases.

Implementation Note
-------------------
Our IMDD core implementation already matches the LMDD scoring mechanics used in
common open-source references, so `vision_lmdd` reuses `CoreIMDD`.
"""

from __future__ import annotations

from typing import Optional

from .baseml import BaseVisionDetector
from .core_feature_base import CoreFeatureDetector
from .imdd import CoreIMDD
from .registry import register_model


@register_model(
    "core_lmdd",
    tags=("classical", "core", "features", "lmdd", "imdd"),
    metadata={
        "description": "LMDD deviation detector for feature matrices (native wrapper)",
        "type": "deviation",
    },
)
class CoreLMDDDetector(CoreFeatureDetector):
    """Feature-matrix LMDD detector (`core_*`).

    Notes
    -----
    LMDD and IMDD share the same scoring mechanics in our implementation,
    so this wrapper reuses :class:`pyimgano.models.imdd.CoreIMDD`.
    """

    def __init__(
        self,
        *,
        contamination: float = 0.1,
        n_iter: int = 50,
        dis_measure: str = "aad",
        random_state: Optional[int] = None,
        **kwargs,
    ) -> None:
        self.n_iter = int(n_iter)
        self.dis_measure = str(dis_measure)
        self.random_state = random_state
        self.kwargs = dict(kwargs)
        super().__init__(contamination=contamination)

    def _build_detector(self):  # noqa: ANN201
        return CoreIMDD(
            contamination=float(self.contamination),
            n_iter=int(self.n_iter),
            dis_measure=str(self.dis_measure),
            random_state=self.random_state,
            **dict(self.kwargs),
        )


@register_model(
    "vision_lmdd",
    tags=("vision", "classical", "lmdd", "baseline"),
    metadata={
        "description": "LMDD deviation detector (native implementation)",
    },
)
class VisionLMDD(BaseVisionDetector):
    """Vision-compatible LMDD detector."""

    def __init__(
        self,
        *,
        feature_extractor=None,
        contamination: float = 0.1,
        n_iter: int = 50,
        dis_measure: str = "aad",
        random_state: Optional[int] = None,
        **kwargs,
    ) -> None:
        self._detector_kwargs = {
            "contamination": float(contamination),
            "n_iter": int(n_iter),
            "dis_measure": str(dis_measure),
            "random_state": random_state,
            **dict(kwargs),
        }
        super().__init__(contamination=contamination, feature_extractor=feature_extractor)

    def _build_detector(self):
        return CoreIMDD(**self._detector_kwargs)
