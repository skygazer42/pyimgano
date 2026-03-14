# -*- coding: utf-8 -*-
"""Cosine kNN detector for embedding feature matrices.

This is a thin, industrial-friendly specialization of `core_knn` that:
- uses cosine distance (good default for normalized deep embeddings)
- optionally L2-normalizes rows internally
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_array

from .core_feature_base import CoreFeatureDetector
from .registry import register_model


class _CosineKNNBackend:
    def __init__(
        self,
        *,
        contamination: float,
        n_neighbors: int,
        method: str,
        normalize: bool,
        eps: float,
        n_jobs: int,
    ) -> None:
        self.contamination = float(contamination)
        self.n_neighbors = int(n_neighbors)
        self.method = str(method)
        self.normalize = bool(normalize)
        self.eps = float(eps)
        self.n_jobs = int(n_jobs)

        self._nn: NearestNeighbors | None = None
        self._k_effective: int | None = None
        self._X_train: np.ndarray | None = None
        self.decision_scores_: np.ndarray | None = None

    def _normalize_rows(self, X: np.ndarray) -> np.ndarray:
        if not self.normalize:
            return np.asarray(X, dtype=np.float64)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.maximum(norms, float(self.eps))
        return np.asarray(X / norms, dtype=np.float64)

    def _aggregate(self, distances: np.ndarray) -> np.ndarray:
        if self.method == "largest":
            return distances.max(axis=1)
        if self.method == "mean":
            return distances.mean(axis=1)
        if self.method == "median":
            return np.median(distances, axis=1)
        raise ValueError("method must be one of {'largest', 'mean', 'median'}")

    def fit(self, X, y=None):  # noqa: ANN001, ANN201 - sklearn-like API
        X = check_array(X, ensure_2d=True, dtype=np.float64)
        X = self._normalize_rows(X)
        self._X_train = X

        n_train = int(X.shape[0])
        if n_train <= 1:
            self._k_effective = 0
            self._nn = None
            self.decision_scores_ = np.zeros(n_train, dtype=np.float64)
            return self

        if self.n_neighbors < 1:
            raise ValueError("n_neighbors must be >= 1")

        k = min(int(self.n_neighbors), n_train - 1)
        self._k_effective = int(k)

        # Cosine is typically implemented via brute-force; keep it explicit to
        # avoid surprising algorithm selection differences across sklearn versions.
        nn = NearestNeighbors(
            n_neighbors=k + 1,
            metric="cosine",
            algorithm="brute",
            n_jobs=int(self.n_jobs),
        )
        nn.fit(X)
        self._nn = nn

        distances, _ = nn.kneighbors(X, n_neighbors=k + 1, return_distance=True)
        distances = np.asarray(distances, dtype=np.float64)[:, 1:]  # drop self
        self.decision_scores_ = self._aggregate(distances).astype(np.float64)
        return self

    def decision_function(self, X):  # noqa: ANN001, ANN201 - sklearn-like API
        if self._X_train is None:
            raise RuntimeError("Detector must be fitted before calling decision_function")
        if self._k_effective is None:
            raise RuntimeError("Internal error: missing k")
        if self._k_effective == 0:
            X_arr = check_array(X, ensure_2d=True, dtype=np.float64)
            return np.zeros(int(X_arr.shape[0]), dtype=np.float64)
        if self._nn is None:
            raise RuntimeError("Internal error: missing neighbor index")

        X_arr = check_array(X, ensure_2d=True, dtype=np.float64)
        X_arr = self._normalize_rows(X_arr)
        distances, _ = self._nn.kneighbors(
            X_arr, n_neighbors=int(self._k_effective), return_distance=True
        )
        distances = np.asarray(distances, dtype=np.float64)
        return self._aggregate(distances).astype(np.float64).ravel()


@register_model(
    "core_knn_cosine",
    tags=("classical", "core", "features", "neighbors", "knn", "cosine"),
    metadata={
        "description": "Cosine kNN distance outlier detector (embedding-friendly)",
        "input": "features",
    },
)
class CoreKNNCosineModel(CoreFeatureDetector):
    """Core cosine-kNN detector with BaseDetector thresholding."""

    def __init__(
        self,
        *,
        contamination: float = 0.1,
        n_neighbors: int = 5,
        method: str = "largest",
        normalize: bool = True,
        eps: float = 1e-12,
        n_jobs: int = 1,
        random_state: Optional[int] = None,  # API compat (unused)
    ) -> None:
        self._backend_kwargs = {
            "contamination": float(contamination),
            "n_neighbors": int(n_neighbors),
            "method": str(method),
            "normalize": bool(normalize),
            "eps": float(eps),
            "n_jobs": int(n_jobs),
        }
        self.random_state = random_state
        super().__init__(contamination=float(contamination))

    def _build_detector(self):
        return _CosineKNNBackend(**self._backend_kwargs)


__all__ = ["CoreKNNCosineModel"]
