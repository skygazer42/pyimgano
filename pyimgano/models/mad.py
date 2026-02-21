# -*- coding: utf-8 -*-
"""
MAD (Median Absolute Deviation) robust baseline.

Unlike PyOD's `MAD` implementation (which is 1D-only and may be sensitive to
`scikit-learn` internal API changes), this implementation is **multivariate**
and works well for image feature vectors (n_samples, n_features).

Scoring:
    z = 0.6745 * |x - median| / MAD

The per-feature robust z-scores are aggregated into a single anomaly score per
sample (e.g., max/mean/l2).
"""

from __future__ import annotations

import logging
from typing import Literal, Optional

import numpy as np
from numpy.typing import NDArray

from .baseml import BaseVisionDetector
from .registry import register_model

logger = logging.getLogger(__name__)


_Aggregation = Literal["max", "mean", "l2"]


class _RobustMADDetector:
    """Small sklearn/PyOD-style detector used by :class:`VisionMAD`."""

    def __init__(
        self,
        *,
        aggregation: _Aggregation = "max",
        eps: float = 1e-12,
        consistency_correction: bool = True,
    ) -> None:
        self.aggregation = str(aggregation).lower()
        self.eps = float(eps)
        self.consistency_correction = bool(consistency_correction)

        self.median_: Optional[NDArray] = None
        self.mad_: Optional[NDArray] = None
        self.decision_scores_: Optional[NDArray] = None

    def fit(self, X: NDArray) -> "_RobustMADDetector":
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim != 2:
            raise ValueError(f"Expected 2D feature matrix, got shape {X_arr.shape}")
        if X_arr.shape[0] == 0:
            raise ValueError("X must contain at least one sample")

        self.median_ = np.median(X_arr, axis=0)
        abs_dev = np.abs(X_arr - self.median_)
        mad = np.median(abs_dev, axis=0)
        mad = np.maximum(mad, self.eps)
        self.mad_ = mad

        self.decision_scores_ = self.decision_function(X_arr)
        return self

    def decision_function(self, X: NDArray) -> NDArray:
        if self.median_ is None or self.mad_ is None:
            raise RuntimeError("Detector not fitted. Call fit() first.")

        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim != 2:
            raise ValueError(f"Expected 2D feature matrix, got shape {X_arr.shape}")
        if X_arr.shape[1] != self.median_.shape[0]:
            raise ValueError(
                "Feature dimension mismatch. "
                f"Expected {self.median_.shape[0]} features, got {X_arr.shape[1]}."
            )

        abs_dev = np.abs(X_arr - self.median_)
        z = abs_dev / self.mad_
        if self.consistency_correction:
            # 0.6745 makes MAD comparable to std-dev for a normal distribution.
            z = 0.6745 * z

        if self.aggregation == "max":
            return np.max(z, axis=1)
        if self.aggregation == "mean":
            return np.mean(z, axis=1)
        if self.aggregation == "l2":
            return np.linalg.norm(z, axis=1)
        raise ValueError(f"Unknown aggregation: {self.aggregation}. Choose from: max, mean, l2")


@register_model(
    "vision_mad",
    tags=("vision", "classical", "mad", "robust", "baseline"),
    metadata={
        "description": "Multivariate MAD robust baseline (median + MAD robust z-score)",
        "type": "robust-statistical",
    },
)
class VisionMAD(BaseVisionDetector):
    """Vision-compatible robust MAD baseline (multivariate)."""

    def __init__(
        self,
        *,
        feature_extractor=None,
        contamination: float = 0.1,
        aggregation: _Aggregation = "max",
        eps: float = 1e-12,
        consistency_correction: bool = True,
    ) -> None:
        if not 0 < float(contamination) < 0.5:
            raise ValueError(
                f"contamination must be in (0, 0.5), got {contamination}"
            )

        self._detector_kwargs = {
            "aggregation": aggregation,
            "eps": float(eps),
            "consistency_correction": bool(consistency_correction),
        }

        logger.debug("Initializing VisionMAD with kwargs=%s", self._detector_kwargs)
        super().__init__(contamination=contamination, feature_extractor=feature_extractor)

    def _build_detector(self):
        return _RobustMADDetector(**self._detector_kwargs)
