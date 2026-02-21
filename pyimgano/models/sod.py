# -*- coding: utf-8 -*-
"""
SOD (Subspace Outlier Detection) wrapper.

Provides PyOD's SOD detector via the unified `pyimgano` vision API.
"""

from __future__ import annotations

import logging

from .baseml import BaseVisionDetector
from .registry import register_model

logger = logging.getLogger(__name__)

try:
    from pyod.models.sod import SOD as _PyODSOD

    _PYOD_AVAILABLE = True
    _IMPORT_ERROR = None
except ImportError as exc:
    _PyODSOD = None
    _PYOD_AVAILABLE = False
    _IMPORT_ERROR = exc


@register_model(
    "vision_sod",
    tags=("vision", "classical", "sod", "subspace", "baseline"),
    metadata={
        "description": "Subspace Outlier Detection via PyOD (subspace baseline)",
        "type": "subspace",
    },
)
class VisionSOD(BaseVisionDetector):
    """Vision-compatible SOD detector (PyOD SOD)."""

    def __init__(
        self,
        *,
        feature_extractor=None,
        contamination: float = 0.1,
        n_neighbors: int = 20,
        ref_set: int = 10,
        alpha: float = 0.8,
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

        n_neighbors_int = int(n_neighbors)
        ref_set_int = int(ref_set)
        if n_neighbors_int < 2:
            raise ValueError(f"n_neighbors must be >= 2. Got {n_neighbors}.")
        if ref_set_int < 1:
            raise ValueError(f"ref_set must be >= 1. Got {ref_set}.")
        if ref_set_int >= n_neighbors_int:
            raise ValueError(
                "ref_set must be < n_neighbors. "
                f"Got ref_set={ref_set_int} and n_neighbors={n_neighbors_int}."
            )

        self._detector_kwargs = {
            "contamination": float(contamination),
            "n_neighbors": n_neighbors_int,
            "ref_set": ref_set_int,
            "alpha": float(alpha),
        }

        logger.debug("Initializing VisionSOD with kwargs=%s", self._detector_kwargs)
        super().__init__(contamination=contamination, feature_extractor=feature_extractor)

    def _build_detector(self):
        return _PyODSOD(**self._detector_kwargs)
