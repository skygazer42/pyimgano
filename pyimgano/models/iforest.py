# -*- coding: utf-8 -*-
"""
IForest (Isolation Forest) wrapper.

This module exposes PyOD's Isolation Forest implementation under the unified
`pyimgano` vision API (image paths -> feature extractor -> 2D feature matrix).
"""

from __future__ import annotations

import logging
from typing import Optional, Union

from .baseml import BaseVisionDetector
from .registry import register_model

logger = logging.getLogger(__name__)

try:
    from pyod.models.iforest import IForest as _PyODIForest

    _PYOD_AVAILABLE = True
    _IMPORT_ERROR = None
except ImportError as exc:
    _PyODIForest = None
    _PYOD_AVAILABLE = False
    _IMPORT_ERROR = exc


@register_model(
    "vision_iforest",
    tags=("vision", "classical", "iforest", "ensemble", "baseline"),
    metadata={
        "description": "Isolation Forest via PyOD (baseline, robust general-purpose)",
        "paper": "Liu et al., Isolation Forest (ICDM 2008)",
        "year": 2008,
        "fast": True,
    },
)
class VisionIForest(BaseVisionDetector):
    """Vision-compatible Isolation Forest (PyOD IForest)."""

    def __init__(
        self,
        *,
        feature_extractor=None,
        contamination: float = 0.1,
        n_estimators: int = 100,
        max_samples: Union[str, int] = "auto",
        max_features: float = 1.0,
        bootstrap: bool = False,
        n_jobs: int = 1,
        random_state: Optional[int] = None,
        verbose: int = 0,
    ) -> None:
        if not _PYOD_AVAILABLE:
            raise ImportError(
                "PyOD is not available. Install it with:\n"
                "  pip install 'pyod>=1.1.0'\n"
                f"Original error: {_IMPORT_ERROR}"
            )

        if not 0 < float(contamination) < 0.5:
            raise ValueError(
                f"contamination must be in (0, 0.5), got {contamination}"
            )

        self._detector_kwargs = {
            "contamination": float(contamination),
            "n_estimators": int(n_estimators),
            "max_samples": max_samples,
            "max_features": float(max_features),
            "bootstrap": bool(bootstrap),
            "n_jobs": int(n_jobs),
            "random_state": random_state,
            "verbose": int(verbose),
        }

        logger.debug("Initializing VisionIForest with kwargs=%s", self._detector_kwargs)
        super().__init__(contamination=contamination, feature_extractor=feature_extractor)

    def _build_detector(self):
        return _PyODIForest(**self._detector_kwargs)
