# -*- coding: utf-8 -*-
"""Robust Z-score detector (median + MAD).

This overlaps conceptually with the MAD baseline but exposes a dedicated core
detector and a different default aggregation.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
from sklearn.utils.validation import check_array

from .base_detector import BaseDetector
from .baseml import BaseVisionDetector
from .registry import register_model
from ..utils.fitted import require_fitted

logger = logging.getLogger(__name__)


_Agg = Literal["mean", "max", "l2"]


@register_model(
    "core_rzscore",
    tags=("classical", "core", "features", "robust", "baseline"),
    metadata={"description": "Robust z-score (median + MAD)"},
)
class CoreRZScore(BaseDetector):
    def __init__(
        self,
        *,
        contamination: float = 0.1,
        aggregation: _Agg = "mean",
        eps: float = 1e-12,
        consistency_correction: bool = True,
    ) -> None:
        super().__init__(contamination=float(contamination))
        self.aggregation = str(aggregation).lower()
        self.eps = float(eps)
        self.consistency_correction = bool(consistency_correction)

    def fit(self, X, y=None):  # noqa: ANN001, ANN201
        X_arr = check_array(X, ensure_2d=True, dtype=np.float64)
        self._set_n_classes(y)
        if X_arr.shape[0] == 0:
            raise ValueError("X must be non-empty")

        self.median_ = np.median(X_arr, axis=0)
        abs_dev = np.abs(X_arr - self.median_)
        mad = np.median(abs_dev, axis=0)
        mad = np.maximum(mad, float(self.eps))
        self.mad_ = mad

        self.decision_scores_ = np.asarray(self.decision_function(X_arr), dtype=np.float64)
        self._process_decision_scores()
        return self

    def decision_function(self, X):  # noqa: ANN001, ANN201
        require_fitted(self, ["median_", "mad_"])
        median = np.asarray(self.median_, dtype=np.float64)  # type: ignore[arg-type]
        mad = np.asarray(self.mad_, dtype=np.float64)  # type: ignore[arg-type]

        X_arr = check_array(X, ensure_2d=True, dtype=np.float64)
        z = np.abs(X_arr - median) / mad
        if self.consistency_correction:
            z = 0.6745 * z

        if self.aggregation == "mean":
            return np.mean(z, axis=1)
        if self.aggregation == "max":
            return np.max(z, axis=1)
        if self.aggregation == "l2":
            return np.linalg.norm(z, axis=1)
        raise ValueError("aggregation must be one of: mean, max, l2")


@register_model(
    "vision_rzscore",
    tags=("vision", "classical", "robust", "baseline"),
    metadata={"description": "Vision robust z-score (median + MAD)"},
)
class VisionRZScore(BaseVisionDetector):
    def __init__(
        self,
        *,
        feature_extractor=None,
        contamination: float = 0.1,
        aggregation: _Agg = "mean",
        eps: float = 1e-12,
        consistency_correction: bool = True,
    ) -> None:
        self._detector_kwargs = {
            "contamination": float(contamination),
            "aggregation": str(aggregation).lower(),
            "eps": float(eps),
            "consistency_correction": bool(consistency_correction),
        }
        logger.debug("Initializing VisionRZScore with kwargs=%s", self._detector_kwargs)
        super().__init__(contamination=contamination, feature_extractor=feature_extractor)

    def _build_detector(self):
        return CoreRZScore(**self._detector_kwargs)
