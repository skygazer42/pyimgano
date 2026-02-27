# -*- coding: utf-8 -*-
"""ODIN (Outlier Detection using Indegree Number).

ODIN builds a kNN graph and uses indegree statistics as an outlier signal.
Points with low indegree are likely outliers.

This implementation supports scoring new samples by measuring the average
indegree of their nearest neighbors in the fitted training graph.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_array

from .base_detector import BaseDetector
from .baseml import BaseVisionDetector
from .registry import register_model
from ..utils.fitted import require_fitted

logger = logging.getLogger(__name__)


@register_model(
    "core_odin",
    tags=("classical", "core", "features", "neighbors", "graph"),
    metadata={"description": "ODIN - indegree-based kNN graph outlier detector (native)"},
)
class CoreODIN(BaseDetector):
    def __init__(
        self,
        *,
        contamination: float = 0.1,
        n_neighbors: int = 10,
        metric: str = "minkowski",
        p: int = 2,
        n_jobs: int | None = None,
        eps: float = 1e-12,
    ) -> None:
        super().__init__(contamination=float(contamination))
        self.n_neighbors = int(n_neighbors)
        self.metric = str(metric)
        self.p = int(p)
        self.n_jobs = n_jobs
        self.eps = float(eps)

    def fit(self, X, y=None):  # noqa: ANN001, ANN201
        X_arr = check_array(X, ensure_2d=True, dtype=np.float64)
        self._set_n_classes(y)

        n = int(X_arr.shape[0])
        k = int(self.n_neighbors)
        if k <= 0:
            raise ValueError(f"n_neighbors must be > 0, got {self.n_neighbors}")
        if n <= k:
            raise ValueError(f"Need n_samples > n_neighbors, got n={n} k={k}")

        nn = NearestNeighbors(
            n_neighbors=k + 1,
            metric=self.metric,
            p=self.p,
            n_jobs=self.n_jobs,
        )
        nn.fit(X_arr)
        _d, indices = nn.kneighbors(X_arr, n_neighbors=k + 1, return_distance=True)
        nbr_idx = np.asarray(indices[:, 1:], dtype=np.int64)

        indegree = np.bincount(nbr_idx.ravel(), minlength=n).astype(np.float64)
        indegree_max = float(np.max(indegree)) if indegree.size else 0.0
        denom = indegree_max + float(self.eps)

        # Low indegree => high score.
        scores = 1.0 - (indegree / denom)

        self._nn = nn
        self._X_train = X_arr
        self.indegree_ = indegree
        self.indegree_max_ = indegree_max

        self.decision_scores_ = np.asarray(scores, dtype=np.float64).reshape(-1)
        self._process_decision_scores()
        return self

    def decision_function(self, X):  # noqa: ANN001, ANN201
        require_fitted(self, ["_nn", "indegree_", "indegree_max_"])
        nn: NearestNeighbors = self._nn  # type: ignore[assignment]
        indegree = np.asarray(self.indegree_, dtype=np.float64).reshape(-1)  # type: ignore[arg-type]
        indegree_max = float(self.indegree_max_)  # type: ignore[arg-type]

        X_arr = check_array(X, ensure_2d=True, dtype=np.float64)
        k = int(self.n_neighbors)

        _d, indices = nn.kneighbors(X_arr, n_neighbors=k, return_distance=True)
        nbr_idx = np.asarray(indices, dtype=np.int64)

        mean_indeg = np.mean(indegree[nbr_idx], axis=1)
        denom = indegree_max + float(self.eps)
        scores = 1.0 - (mean_indeg / denom)
        return np.asarray(scores, dtype=np.float64).reshape(-1)


@register_model(
    "vision_odin",
    tags=("vision", "classical", "neighbors", "graph"),
    metadata={"description": "Vision ODIN - indegree-based kNN graph detector"},
)
class VisionODIN(BaseVisionDetector):
    def __init__(
        self,
        *,
        feature_extractor=None,
        contamination: float = 0.1,
        n_neighbors: int = 10,
        metric: str = "minkowski",
        p: int = 2,
        n_jobs: int | None = None,
        eps: float = 1e-12,
    ) -> None:
        self._detector_kwargs = {
            "contamination": float(contamination),
            "n_neighbors": int(n_neighbors),
            "metric": str(metric),
            "p": int(p),
            "n_jobs": n_jobs,
            "eps": float(eps),
        }
        logger.debug("Initializing VisionODIN with kwargs=%s", self._detector_kwargs)
        super().__init__(contamination=contamination, feature_extractor=feature_extractor)

    def _build_detector(self):
        return CoreODIN(**self._detector_kwargs)
