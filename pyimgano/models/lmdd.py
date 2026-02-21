# -*- coding: utf-8 -*-
"""
LMDD wrapper.

Provides PyOD's LMDD detector via the unified `pyimgano` vision API.
"""

from __future__ import annotations

import logging
from typing import Optional

from .baseml import BaseVisionDetector
from .registry import register_model

logger = logging.getLogger(__name__)

try:
    from pyod.models.lmdd import LMDD as _PyODLMDD

    _PYOD_AVAILABLE = True
    _IMPORT_ERROR = None
except ImportError as exc:
    _PyODLMDD = None
    _PYOD_AVAILABLE = False
    _IMPORT_ERROR = exc


@register_model(
    "vision_lmdd",
    tags=("vision", "classical", "lmdd", "baseline"),
    metadata={
        "description": "LMDD via PyOD (baseline)",
    },
)
class VisionLMDD(BaseVisionDetector):
    """Vision-compatible LMDD detector (PyOD LMDD)."""

    def __init__(
        self,
        *,
        feature_extractor=None,
        contamination: float = 0.1,
        n_iter: int = 50,
        dis_measure: str = "aad",
        random_state: Optional[int] = None,
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
            "n_iter": int(n_iter),
            "dis_measure": str(dis_measure),
            "random_state": random_state,
        }

        logger.debug("Initializing VisionLMDD with kwargs=%s", self._detector_kwargs)
        super().__init__(contamination=contamination, feature_extractor=feature_extractor)

    def _build_detector(self):
        return _PyODLMDD(**self._detector_kwargs)

