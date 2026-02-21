# -*- coding: utf-8 -*-
"""
KDE (Kernel Density Estimation) wrapper.

Provides PyOD's KDE detector via the unified `pyimgano` vision API.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from .baseml import BaseVisionDetector
from .registry import register_model

logger = logging.getLogger(__name__)

try:
    from pyod.models.kde import KDE as _PyODKDE

    _PYOD_AVAILABLE = True
    _IMPORT_ERROR = None
except ImportError as exc:
    _PyODKDE = None
    _PYOD_AVAILABLE = False
    _IMPORT_ERROR = exc


@register_model(
    "vision_kde",
    tags=("vision", "classical", "kde", "density", "baseline"),
    metadata={
        "description": "Kernel Density Estimation via PyOD (density baseline)",
        "type": "density",
    },
)
class VisionKDE(BaseVisionDetector):
    """Vision-compatible KDE detector (PyOD KDE)."""

    def __init__(
        self,
        *,
        feature_extractor=None,
        contamination: float = 0.1,
        bandwidth: float = 1.0,
        algorithm: str = "auto",
        leaf_size: int = 30,
        metric: str = "minkowski",
        metric_params: Optional[dict[str, Any]] = None,
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
            "bandwidth": float(bandwidth),
            "algorithm": str(algorithm),
            "leaf_size": int(leaf_size),
            "metric": str(metric),
            "metric_params": dict(metric_params) if metric_params is not None else None,
        }

        logger.debug("Initializing VisionKDE with kwargs=%s", self._detector_kwargs)
        super().__init__(contamination=contamination, feature_extractor=feature_extractor)

    def _build_detector(self):
        return _PyODKDE(**self._detector_kwargs)

