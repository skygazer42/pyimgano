# -*- coding: utf-8 -*-
"""Mahalanobis distance with covariance shrinkage (Ledoit-Wolf).

For deep embeddings, naive covariance can be ill-conditioned when:
- feature dim is high
- sample count is modest

Ledoit-Wolf shrinkage is a practical industrial default: stable and fast.
"""

from __future__ import annotations

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.utils import check_array

from .core_feature_base import CoreFeatureDetector
from .registry import register_model


class _MahalanobisShrinkageBackend:
    def __init__(self, *, assume_centered: bool) -> None:
        self.assume_centered = bool(assume_centered)
        self._lw: LedoitWolf | None = None
        self.decision_scores_: np.ndarray | None = None

    def fit(self, X, y=None):  # noqa: ANN001, ANN201 - sklearn-like API
        X_arr = check_array(X, ensure_2d=True, dtype=np.float64)
        if int(X_arr.shape[0]) == 0:
            raise ValueError("Training set cannot be empty")

        lw = LedoitWolf(assume_centered=bool(self.assume_centered))
        lw.fit(X_arr)
        self._lw = lw
        self.decision_scores_ = np.asarray(self.decision_function(X_arr), dtype=np.float64).reshape(
            -1
        )
        return self

    def decision_function(self, X):  # noqa: ANN001, ANN201 - sklearn-like API
        if self._lw is None:
            raise RuntimeError("Detector must be fitted before calling decision_function")
        X_arr = check_array(X, ensure_2d=True, dtype=np.float64)
        # sklearn returns squared Mahalanobis distances.
        scores = self._lw.mahalanobis(X_arr)
        return np.asarray(scores, dtype=np.float64).reshape(-1)


@register_model(
    "core_mahalanobis_shrinkage",
    tags=("classical", "core", "features", "distance", "gaussian", "shrinkage"),
    metadata={
        "description": "Mahalanobis distance with Ledoit-Wolf covariance shrinkage (embedding-friendly)",
        "input": "features",
    },
)
class CoreMahalanobisShrinkage(CoreFeatureDetector):
    def __init__(
        self,
        *,
        contamination: float = 0.1,
        assume_centered: bool = False,
    ) -> None:
        self.assume_centered = bool(assume_centered)
        super().__init__(contamination=float(contamination))

    def _build_detector(self):
        return _MahalanobisShrinkageBackend(assume_centered=bool(self.assume_centered))


__all__ = ["CoreMahalanobisShrinkage"]
