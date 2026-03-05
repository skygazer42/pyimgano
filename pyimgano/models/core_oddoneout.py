# -*- coding: utf-8 -*-
"""Odd-One-Out (neighbor comparison) core detector for feature matrices.

This implements a lightweight, industrial-friendly variant inspired by the
CVPR 2025 "Odd-One-Out" neighbor comparison idea:

- Build a kNN index on normal (training) features
- Score each sample by *relative* neighborhood distance:

    score(x) = mean_k(dist(x, NN_k(train))) / mean_k(train_local_mean[NN_k(train)])

Intuition:
- A point is anomalous if it is further from its neighbors than those neighbors
  are from *their* neighbors (i.e., it is the "odd one out" locally).

Design goals:
- `core_*` contract: accepts `np.ndarray` / torch tensors (via CoreFeatureDetector)
- Higher score => more anomalous
- Safe-by-default (no deep backbones; no downloads)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_array

from .core_feature_base import CoreFeatureDetector
from .registry import register_model


class _OddOneOutBackend:
    def __init__(
        self,
        *,
        contamination: float,
        n_neighbors: int,
        metric: str,
        p: int,
        method: str,
        normalize: bool,
        eps: float,
        n_jobs: int,
        random_state: Optional[int],
    ) -> None:
        self.contamination = float(contamination)
        self.n_neighbors = int(n_neighbors)
        self.metric = str(metric)
        self.p = int(p)
        self.method = str(method)
        self.normalize = bool(normalize)
        self.eps = float(eps)
        self.n_jobs = int(n_jobs)
        self.random_state = random_state

        self._nn: NearestNeighbors | None = None
        self._k_effective: int | None = None
        self._train_local_mean: np.ndarray | None = None
        self.decision_scores_: np.ndarray | None = None

    def _normalize_rows(self, X: np.ndarray) -> np.ndarray:
        if not self.normalize:
            return np.asarray(X, dtype=np.float64)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.maximum(norms, float(self.eps))
        return np.asarray(X / norms, dtype=np.float64)

    def _aggregate(self, distances: np.ndarray) -> np.ndarray:
        if self.method == "mean":
            return distances.mean(axis=1)
        if self.method == "median":
            return np.median(distances, axis=1)
        if self.method == "largest":
            return distances.max(axis=1)
        raise ValueError("method must be one of {'mean', 'median', 'largest'}")

    def fit(self, X, y=None):  # noqa: ANN001, ANN201 - sklearn-like API
        X = check_array(X, ensure_2d=True, dtype=np.float64)
        X = self._normalize_rows(X)

        n_train = int(X.shape[0])
        if n_train <= 1:
            self._k_effective = 0
            self._nn = None
            self._train_local_mean = np.zeros(n_train, dtype=np.float64)
            self.decision_scores_ = np.zeros(n_train, dtype=np.float64)
            return self

        if self.n_neighbors < 1:
            raise ValueError("n_neighbors must be >= 1")

        k = min(int(self.n_neighbors), n_train - 1)
        self._k_effective = int(k)

        algo = "brute" if self.metric == "cosine" else "auto"
        nn = NearestNeighbors(
            n_neighbors=k + 1,  # include self for training
            metric=self.metric,
            p=self.p,
            algorithm=algo,
            n_jobs=int(self.n_jobs),
        )
        nn.fit(X)
        self._nn = nn

        dist, idx = nn.kneighbors(X, n_neighbors=k + 1, return_distance=True)
        dist = np.asarray(dist, dtype=np.float64)[:, 1:]  # drop self
        idx = np.asarray(idx, dtype=np.int64)[:, 1:]

        local_mean = self._aggregate(dist).astype(np.float64)
        self._train_local_mean = local_mean

        # Relative "oddness": compare sample neighborhood distance to the neighbors' typical distance.
        neigh_baseline = np.mean(local_mean[idx], axis=1)
        neigh_baseline = np.maximum(neigh_baseline, float(self.eps))
        scores = local_mean / neigh_baseline
        self.decision_scores_ = np.asarray(scores, dtype=np.float64).reshape(-1)
        return self

    def decision_function(self, X):  # noqa: ANN001, ANN201 - sklearn-like API
        if self._k_effective is None or self._train_local_mean is None:
            raise RuntimeError("Detector must be fitted before calling decision_function")
        k = int(self._k_effective)
        X_arr = check_array(X, ensure_2d=True, dtype=np.float64)
        if k <= 0:
            return np.zeros(int(X_arr.shape[0]), dtype=np.float64)
        if self._nn is None:
            raise RuntimeError("Internal error: missing neighbor index")

        X_arr = self._normalize_rows(X_arr)
        dist, idx = self._nn.kneighbors(X_arr, n_neighbors=k, return_distance=True)
        dist = np.asarray(dist, dtype=np.float64)
        idx = np.asarray(idx, dtype=np.int64)

        local_mean = self._aggregate(dist).astype(np.float64)
        neigh_baseline = np.mean(np.asarray(self._train_local_mean, dtype=np.float64)[idx], axis=1)
        neigh_baseline = np.maximum(neigh_baseline, float(self.eps))
        scores = local_mean / neigh_baseline
        return np.asarray(scores, dtype=np.float64).reshape(-1)


@register_model(
    "core_oddoneout",
    tags=("classical", "core", "features", "neighbors", "oddoneout", "cvpr2025"),
    metadata={
        "description": "Odd-One-Out (neighbor comparison) core detector on feature matrices",
        "input": "features",
    },
)
class CoreOddOneOut(CoreFeatureDetector):
    """Odd-One-Out neighbor comparison detector for `np.ndarray` / torch feature matrices."""

    def __init__(
        self,
        *,
        contamination: float = 0.1,
        n_neighbors: int = 5,
        metric: str = "minkowski",
        p: int = 2,
        method: str = "mean",
        normalize: bool = True,
        eps: float = 1e-12,
        n_jobs: int = 1,
        random_state: Optional[int] = 0,
    ) -> None:
        self._backend_kwargs = dict(
            contamination=float(contamination),
            n_neighbors=int(n_neighbors),
            metric=str(metric),
            p=int(p),
            method=str(method),
            normalize=bool(normalize),
            eps=float(eps),
            n_jobs=int(n_jobs),
            random_state=random_state,
        )
        self.random_state = random_state
        super().__init__(contamination=float(contamination))

    def _build_detector(self):
        return _OddOneOutBackend(**self._backend_kwargs)


__all__ = ["CoreOddOneOut"]
