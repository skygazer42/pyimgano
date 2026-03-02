# -*- coding: utf-8 -*-
"""Cosine-style Mahalanobis distance on embedding feature matrices.

This model is an embedding-friendly variant of Mahalanobis distance:
- (optional) L2-normalize each feature vector (direction matters; magnitude less so)
- fit a Gaussian model with Ledoit-Wolf covariance shrinkage
- score via (squared) Mahalanobis distance

Score convention: **higher score => more anomalous**.
"""

from __future__ import annotations

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.utils import check_array

from .core_feature_base import CoreFeatureDetector
from .registry import register_model


class _CosineMahalanobisBackend:
    def __init__(
        self,
        *,
        assume_centered: bool,
        normalize: bool,
        eps: float,
    ) -> None:
        self.assume_centered = bool(assume_centered)
        self.normalize = bool(normalize)
        self.eps = float(eps)

        self._lw: LedoitWolf | None = None
        self.decision_scores_: np.ndarray | None = None

    def _normalize_rows(self, X: np.ndarray) -> np.ndarray:
        if not self.normalize:
            return np.asarray(X, dtype=np.float64)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.maximum(norms, float(self.eps))
        return np.asarray(X / norms, dtype=np.float64)

    def fit(self, X, y=None):  # noqa: ANN001, ANN201 - sklearn-like API
        X_arr = check_array(X, ensure_2d=True, dtype=np.float64)
        if int(X_arr.shape[0]) == 0:
            raise ValueError("Training set cannot be empty")

        Z = self._normalize_rows(X_arr)
        lw = LedoitWolf(assume_centered=bool(self.assume_centered))
        lw.fit(Z)
        self._lw = lw
        self.decision_scores_ = np.asarray(self.decision_function(Z), dtype=np.float64).reshape(-1)
        return self

    def decision_function(self, X):  # noqa: ANN001, ANN201 - sklearn-like API
        if self._lw is None:
            raise RuntimeError("Detector must be fitted before calling decision_function")
        X_arr = check_array(X, ensure_2d=True, dtype=np.float64)
        Z = self._normalize_rows(X_arr)
        # sklearn returns squared Mahalanobis distances.
        scores = self._lw.mahalanobis(Z)
        return np.asarray(scores, dtype=np.float64).reshape(-1)


@register_model(
    "core_cosine_mahalanobis",
    tags=("classical", "core", "features", "distance", "gaussian", "shrinkage", "cosine"),
    metadata={
        "description": "Mahalanobis distance on L2-normalized embeddings with Ledoit-Wolf covariance shrinkage",
        "input": "features",
    },
)
class CoreCosineMahalanobis(CoreFeatureDetector):
    def __init__(
        self,
        *,
        contamination: float = 0.1,
        assume_centered: bool = False,
        normalize: bool = True,
        eps: float = 1e-12,
    ) -> None:
        self.assume_centered = bool(assume_centered)
        self.normalize = bool(normalize)
        self.eps = float(eps)
        super().__init__(contamination=float(contamination))

    def _build_detector(self):
        return _CosineMahalanobisBackend(
            assume_centered=bool(self.assume_centered),
            normalize=bool(self.normalize),
            eps=float(self.eps),
        )


__all__ = ["CoreCosineMahalanobis"]

