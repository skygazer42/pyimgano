# -*- coding: utf-8 -*-
"""Neighborhood entropy baseline (kNN distance distribution).

We compute a simple entropy-based statistic over kNN distance weights.
This is a lightweight graph-inspired score useful as an additional baseline.

Algorithm
---------
For each sample:
1) Compute kNN distances d_1..d_k
2) Convert to weights w_i = exp(-d_i / scale)
3) Normalize p_i = w_i / sum(w)
4) Entropy H = -sum p_i log p_i
5) Score = 1 - H / log(k)  (higher => more anomalous)
"""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_array

from pyimgano.utils.fitted import require_fitted

from .base_detector import BaseDetector
from .baseml import BaseVisionDetector
from .registry import register_model


def _entropy_scores(dist: np.ndarray, *, eps: float) -> np.ndarray:
    d = np.asarray(dist, dtype=np.float64)
    if d.ndim != 2:
        raise ValueError(f"Expected distances shape (N,k), got {d.shape}")
    n, k = int(d.shape[0]), int(d.shape[1])
    if n == 0:
        return np.zeros((0,), dtype=np.float64)
    if k <= 1:
        return np.zeros((n,), dtype=np.float64)

    scale = np.median(d, axis=1, keepdims=True)
    scale = np.maximum(scale, float(eps))
    w = np.exp(-d / scale)
    denom = np.sum(w, axis=1, keepdims=True)
    denom = np.maximum(denom, float(eps))
    p = w / denom

    p = np.maximum(p, float(eps))
    H = -np.sum(p * np.log(p), axis=1)
    Hn = H / max(np.log(float(k)), float(eps))
    score = 1.0 - Hn
    return np.asarray(score, dtype=np.float64).reshape(-1)


@register_model(
    "core_neighborhood_entropy",
    tags=("classical", "core", "features", "neighbors", "graph", "entropy"),
    metadata={
        "description": "Neighborhood entropy score over kNN distances (native baseline)",
        "type": "neighbors",
    },
)
class CoreNeighborhoodEntropy(BaseDetector):
    def __init__(
        self,
        *,
        contamination: float = 0.1,
        n_neighbors: int = 20,
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
            raise ValueError("n_neighbors must be > 0")
        if n <= 1:
            self._X_train = X_arr
            self._nn = NearestNeighbors(n_neighbors=1, metric=self.metric).fit(X_arr)
            self.decision_scores_ = np.zeros((n,), dtype=np.float64)
            self._process_decision_scores()
            return self

        k_eff = min(k, n - 1)
        nn = NearestNeighbors(
            n_neighbors=k_eff + 1,
            metric=self.metric,
            p=self.p,
            n_jobs=self.n_jobs,
        )
        nn.fit(X_arr)
        dist, _idx = nn.kneighbors(X_arr, n_neighbors=k_eff + 1, return_distance=True)
        dist = np.asarray(dist[:, 1:], dtype=np.float64)

        self._X_train = X_arr
        self._nn = nn
        self._k_eff = int(k_eff)

        self.decision_scores_ = _entropy_scores(dist, eps=float(self.eps))
        self._process_decision_scores()
        return self

    def decision_function(self, X):  # noqa: ANN001, ANN201
        require_fitted(self, ["_nn", "_X_train", "_k_eff"])
        X_arr = check_array(X, ensure_2d=True, dtype=np.float64)
        nn: NearestNeighbors = self._nn  # type: ignore[assignment]
        k_eff = int(self._k_eff)  # type: ignore[arg-type]
        if k_eff <= 0:
            return np.zeros((X_arr.shape[0],), dtype=np.float64)
        dist, _idx = nn.kneighbors(X_arr, n_neighbors=k_eff, return_distance=True)
        return _entropy_scores(dist, eps=float(self.eps))


@register_model(
    "vision_neighborhood_entropy",
    tags=("vision", "classical", "neighbors", "graph", "entropy"),
    metadata={"description": "Vision wrapper for neighborhood entropy score"},
)
class VisionNeighborhoodEntropy(BaseVisionDetector):
    def __init__(
        self,
        *,
        feature_extractor=None,
        contamination: float = 0.1,
        n_neighbors: int = 20,
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
        super().__init__(contamination=contamination, feature_extractor=feature_extractor)

    def _build_detector(self):
        return CoreNeighborhoodEntropy(**self._detector_kwargs)

