# -*- coding: utf-8 -*-
"""Local Outlier Factor (LOF) core implementation.

This module provides:
- `CoreLOF`: a small sklearn-style backend on feature matrices
- `core_lof`: a registry entry wrapped with `CoreFeatureDetector` thresholding

Notes
-----
scikit-learn's LOF scoring convention is "higher => more normal". We negate
scores to match PyImgAno's convention: **higher score => more anomalous**.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.utils import check_array

from .core_feature_base import CoreFeatureDetector
from .registry import register_model


class CoreLOF:
    """Sklearn-backed LOF core in novelty mode (scores new samples)."""

    def __init__(
        self,
        *,
        contamination: float = 0.1,
        n_neighbors: int = 20,
        metric: str = "minkowski",
        p: int = 2,
        leaf_size: int = 30,
        n_jobs: Optional[int] = None,
    ) -> None:
        self.contamination = float(contamination)
        self.n_neighbors = int(n_neighbors)
        self.metric = str(metric)
        self.p = int(p)
        self.leaf_size = int(leaf_size)
        self.n_jobs = n_jobs

        self.detector_: LocalOutlierFactor | None = None
        self.decision_scores_: np.ndarray | None = None

    def fit(self, X, y=None):  # noqa: ANN001, ANN201 - sklearn-like API
        X = check_array(X, ensure_2d=True, dtype=np.float64)
        n = int(X.shape[0])
        if n == 0:
            raise ValueError("Training set cannot be empty")
        if n <= 2:
            self.detector_ = None
            self.decision_scores_ = np.zeros((n,), dtype=np.float64)
            return self

        k = int(self.n_neighbors)
        if k < 1:
            raise ValueError("n_neighbors must be >= 1")
        k = min(k, n - 1)

        # We threshold outside via BaseDetector; keep sklearn contamination out of the way.
        det = LocalOutlierFactor(
            n_neighbors=k,
            novelty=True,
            contamination="auto",
            metric=self.metric,
            p=self.p,
            leaf_size=self.leaf_size,
            n_jobs=self.n_jobs,
        )
        det.fit(X)
        self.detector_ = det

        # sklearn: more negative => more abnormal. Negate to match "higher => more anomalous".
        self.decision_scores_ = (
            -np.asarray(det.negative_outlier_factor_, dtype=np.float64)
        ).reshape(-1)
        return self

    def decision_function(self, X):  # noqa: ANN001, ANN201 - sklearn-like API
        if self.decision_scores_ is None:
            raise RuntimeError("Detector must be fitted before calling decision_function")

        X = check_array(X, ensure_2d=True, dtype=np.float64)
        if self.detector_ is None:
            return np.zeros((X.shape[0],), dtype=np.float64)

        # sklearn: score_samples higher => inlier. Negate to match "higher => more anomalous".
        return (-np.asarray(self.detector_.score_samples(X), dtype=np.float64)).reshape(-1)


@register_model(
    "core_lof",
    tags=("classical", "core", "features", "lof", "neighbors", "density"),
    metadata={
        "description": "Core Local Outlier Factor detector on feature matrices (sklearn backend)",
        "input": "features",
        "paper": "Breunig et al., SIGMOD 2000",
        "year": 2000,
    },
)
class CoreLOFModel(CoreFeatureDetector):
    """Core (feature-matrix) LOF detector with BaseDetector thresholding."""

    def __init__(
        self,
        *,
        contamination: float = 0.1,
        n_neighbors: int = 20,
        metric: str = "minkowski",
        p: int = 2,
        leaf_size: int = 30,
        n_jobs: Optional[int] = None,
    ) -> None:
        self._backend_kwargs = {
            "contamination": float(contamination),
            "n_neighbors": int(n_neighbors),
            "metric": str(metric),
            "p": int(p),
            "leaf_size": int(leaf_size),
            "n_jobs": n_jobs,
        }
        super().__init__(contamination=float(contamination))

    def _build_detector(self):
        return CoreLOF(**self._backend_kwargs)
