# -*- coding: utf-8 -*-
"""
QMCD (Quantile-based Minimum Covariance Determinant) wrapper.

Provides PyOD's QMCD detector via the unified `pyimgano` vision API.
"""

from __future__ import annotations

import logging

from .baseml import BaseVisionDetector
from .registry import register_model

logger = logging.getLogger(__name__)

try:
    from pyod.models.qmcd import QMCD as _PyODQMCD

    _PYOD_AVAILABLE = True
    _IMPORT_ERROR = None
except ImportError as exc:
    _PyODQMCD = None
    _PYOD_AVAILABLE = False
    _IMPORT_ERROR = exc


@register_model(
    "vision_qmcd",
    tags=("vision", "classical", "qmcd", "robust", "baseline"),
    metadata={
        "description": "QMCD via PyOD (robust covariance baseline)",
        "type": "robust-statistical",
    },
)
class VisionQMCD(BaseVisionDetector):
    """Vision-compatible QMCD detector (PyOD QMCD)."""

    def __init__(
        self,
        *,
        feature_extractor=None,
        contamination: float = 0.1,
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

        self._detector_kwargs = {"contamination": float(contamination)}

        logger.debug("Initializing VisionQMCD with kwargs=%s", self._detector_kwargs)
        super().__init__(contamination=contamination, feature_extractor=feature_extractor)

    def _build_detector(self):
        return _PyODQMCD(**self._detector_kwargs)

