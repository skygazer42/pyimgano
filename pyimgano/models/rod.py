# -*- coding: utf-8 -*-
"""
ROD (Rotation-based Outlier Detection) wrapper.

Provides PyOD's ROD detector via the unified `pyimgano` vision API.
"""

from __future__ import annotations

import logging

from .baseml import BaseVisionDetector
from .registry import register_model

logger = logging.getLogger(__name__)

try:
    from pyod.models.rod import ROD as _PyODROD

    _PYOD_AVAILABLE = True
    _IMPORT_ERROR = None
except ImportError as exc:
    _PyODROD = None
    _PYOD_AVAILABLE = False
    _IMPORT_ERROR = exc


@register_model(
    "vision_rod",
    tags=("vision", "classical", "rod", "baseline"),
    metadata={
        "description": "Rotation-based Outlier Detection via PyOD (baseline)",
    },
)
class VisionROD(BaseVisionDetector):
    """Vision-compatible ROD detector (PyOD ROD)."""

    def __init__(
        self,
        *,
        feature_extractor=None,
        contamination: float = 0.1,
        parallel_execution: bool = False,
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
            "parallel_execution": bool(parallel_execution),
        }

        logger.debug("Initializing VisionROD with kwargs=%s", self._detector_kwargs)
        super().__init__(contamination=contamination, feature_extractor=feature_extractor)

    def _build_detector(self):
        return _PyODROD(**self._detector_kwargs)

