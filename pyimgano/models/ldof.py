# -*- coding: utf-8 -*-
"""LDOF (Local Distance-based Outlier Factor).

Score definition (kNN-based):
    d_in(i)  = mean distance from x_i to its k nearest neighbors
    d_out(i) = mean pairwise distance among those k neighbors
    ldof(i)  = d_in(i) / (d_out(i) + eps)

Higher scores indicate a point is farther from its neighbors than the neighbors
are from each other.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_array

from ..utils.fitted import require_fitted
from .base_detector import BaseDetector
from .baseml import BaseVisionDetector
from .registry import register_model

logger = logging.getLogger(__name__)


def _mean_pairwise_distance(points_2d: np.ndarray) -> float:
    pts = np.asarray(points_2d, dtype=np.float64)
    if pts.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {pts.shape}")
    k = int(pts.shape[0])
    if k <= 1:
        return 0.0
    # Compute Euclidean distances via Gram matrix.
    sq = np.sum(pts * pts, axis=1)
    g = pts @ pts.T
    d2 = sq[:, None] + sq[None, :] - 2.0 * g
    d2 = np.maximum(d2, 0.0)
    # Upper triangle without diagonal.
    iu = np.triu_indices(k, k=1)
    d = np.sqrt(d2[iu])
    return float(np.mean(d)) if d.size else 0.0


@register_model(
    "core_ldof",
    tags=("classical", "core", "features", "neighbors", "local"),
    metadata={
        "description": "LDOF - Local Distance-based Outlier Factor (native)",
        "paper": "Zhang et al., PAKDD 2009",
        "year": 2009,
    },
)
class CoreLDOF(BaseDetector):
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

    def fit(self, x, y=None):  # noqa: ANN001, ANN201
        x_arr = check_array(x, ensure_2d=True, dtype=np.float64)
        self._set_n_classes(y)

        n = int(x_arr.shape[0])
        k = int(self.n_neighbors)
        if k <= 1:
            raise ValueError(f"n_neighbors must be >= 2, got {self.n_neighbors}")
        if n <= k:
            raise ValueError(f"Need n_samples > n_neighbors, got n={n} k={k}")

        nn = NearestNeighbors(
            n_neighbors=k + 1,
            metric=self.metric,
            p=self.p,
            n_jobs=self.n_jobs,
        )
        nn.fit(x_arr)
        distances, indices = nn.kneighbors(x_arr, n_neighbors=k + 1, return_distance=True)
        d = np.asarray(distances[:, 1:], dtype=np.float64)
        nbr_idx = np.asarray(indices[:, 1:], dtype=np.int64)

        d_in = np.mean(d, axis=1)
        d_out = np.empty(n, dtype=np.float64)
        for i in range(n):
            pts = x_arr[nbr_idx[i]]
            d_out[i] = _mean_pairwise_distance(pts)

        scores = d_in / (d_out + float(self.eps))
        self._nn = nn
        self._X_train = x_arr
        self.decision_scores_ = np.asarray(scores, dtype=np.float64).reshape(-1)
        self._process_decision_scores()
        return self

    def decision_function(self, x):  # noqa: ANN001, ANN201
        require_fitted(self, ["_nn", "_X_train"])
        nn: NearestNeighbors = self._nn  # type: ignore[assignment]
        x_train = np.asarray(self._X_train, dtype=np.float64)  # type: ignore[arg-type]

        x_arr = check_array(x, ensure_2d=True, dtype=np.float64)
        k = int(self.n_neighbors)

        distances, indices = nn.kneighbors(x_arr, n_neighbors=k, return_distance=True)
        d = np.asarray(distances, dtype=np.float64)
        nbr_idx = np.asarray(indices, dtype=np.int64)

        d_in = np.mean(d, axis=1)
        d_out = np.empty(x_arr.shape[0], dtype=np.float64)
        for i in range(x_arr.shape[0]):
            pts = x_train[nbr_idx[i]]
            d_out[i] = _mean_pairwise_distance(pts)

        scores = d_in / (d_out + float(self.eps))
        return np.asarray(scores, dtype=np.float64).reshape(-1)


@register_model(
    "vision_ldof",
    tags=("vision", "classical", "neighbors", "local"),
    metadata={
        "description": "Vision LDOF - Local Distance-based Outlier Factor",
        "paper": "Zhang et al., PAKDD 2009",
        "year": 2009,
    },
)
class VisionLDOF(BaseVisionDetector):
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
        logger.debug("Initializing VisionLDOF with kwargs=%s", self._detector_kwargs)
        super().__init__(contamination=contamination, feature_extractor=feature_extractor)

    def _build_detector(self):
        return CoreLDOF(**self._detector_kwargs)
