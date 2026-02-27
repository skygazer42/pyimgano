# -*- coding: utf-8 -*-
"""Distance-to-centroid (DTC) baseline.

Score = ||x - mean||_2
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.utils.validation import check_array

from .base_detector import BaseDetector
from .baseml import BaseVisionDetector
from .registry import register_model
from ..utils.fitted import require_fitted

logger = logging.getLogger(__name__)


@register_model(
    "core_dtc",
    tags=("classical", "core", "features", "distance", "baseline"),
    metadata={"description": "Distance-to-centroid baseline (L2 to mean)"},
)
class CoreDTC(BaseDetector):
    def __init__(self, *, contamination: float = 0.1) -> None:
        super().__init__(contamination=float(contamination))

    def fit(self, X, y=None):  # noqa: ANN001, ANN201
        X_arr = check_array(X, ensure_2d=True, dtype=np.float64)
        self._set_n_classes(y)
        if X_arr.shape[0] == 0:
            raise ValueError("X must be non-empty")
        self.mean_ = np.mean(X_arr, axis=0)
        self.decision_scores_ = np.asarray(self.decision_function(X_arr), dtype=np.float64)
        self._process_decision_scores()
        return self

    def decision_function(self, X):  # noqa: ANN001, ANN201
        require_fitted(self, ["mean_"])
        mu = np.asarray(self.mean_, dtype=np.float64)  # type: ignore[arg-type]
        X_arr = check_array(X, ensure_2d=True, dtype=np.float64)
        diff = X_arr - mu
        return np.linalg.norm(diff, axis=1)


@register_model(
    "vision_dtc",
    tags=("vision", "classical", "distance", "baseline"),
    metadata={"description": "Vision distance-to-centroid baseline"},
)
class VisionDTC(BaseVisionDetector):
    def __init__(self, *, feature_extractor=None, contamination: float = 0.1) -> None:
        self._detector_kwargs = {"contamination": float(contamination)}
        logger.debug("Initializing VisionDTC with kwargs=%s", self._detector_kwargs)
        super().__init__(contamination=contamination, feature_extractor=feature_extractor)

    def _build_detector(self):
        return CoreDTC(**self._detector_kwargs)
