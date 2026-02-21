# -*- coding: utf-8 -*-
"""
GMM (Gaussian Mixture Model) wrapper.

Exposes PyOD's GMM detector via the unified `pyimgano` vision API.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from .baseml import BaseVisionDetector
from .registry import register_model

logger = logging.getLogger(__name__)

try:
    from pyod.models.gmm import GMM as _PyODGMM

    _PYOD_AVAILABLE = True
    _IMPORT_ERROR = None
except ImportError as exc:
    _PyODGMM = None
    _PYOD_AVAILABLE = False
    _IMPORT_ERROR = exc


@register_model(
    "vision_gmm",
    tags=("vision", "classical", "gmm", "density", "baseline"),
    metadata={
        "description": "Gaussian Mixture Model via PyOD (density baseline)",
        "type": "density",
    },
)
class VisionGMM(BaseVisionDetector):
    """Vision-compatible GMM detector (PyOD GMM)."""

    def __init__(
        self,
        *,
        feature_extractor=None,
        contamination: float = 0.1,
        n_components: int = 1,
        covariance_type: str = "full",
        tol: float = 1e-3,
        reg_covar: float = 1e-6,
        max_iter: int = 100,
        n_init: int = 1,
        init_params: str = "kmeans",
        weights_init=None,
        means_init=None,
        precisions_init=None,
        random_state: Optional[int] = None,
        warm_start: bool = False,
        **kwargs: Any,
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

        # Keep kwargs passthrough for forward compatibility with PyOD/sklearn.
        self._detector_kwargs = {
            "contamination": float(contamination),
            "n_components": int(n_components),
            "covariance_type": str(covariance_type),
            "tol": float(tol),
            "reg_covar": float(reg_covar),
            "max_iter": int(max_iter),
            "n_init": int(n_init),
            "init_params": str(init_params),
            "weights_init": weights_init,
            "means_init": means_init,
            "precisions_init": precisions_init,
            "random_state": random_state,
            "warm_start": bool(warm_start),
            **dict(kwargs),
        }

        logger.debug("Initializing VisionGMM with kwargs=%s", self._detector_kwargs)
        super().__init__(contamination=contamination, feature_extractor=feature_extractor)

    def _build_detector(self):
        return _PyODGMM(**self._detector_kwargs)

