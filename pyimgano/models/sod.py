# -*- coding: utf-8 -*-
"""
SOD (Subspace Outlier Detection).

SOD detects outliers in varying subspaces by comparing each point against a
reference set derived from shared nearest neighbors (SNN).

Reference:
    Kriegel, H.P., Kröger, P., Schubert, E. and Zimek, A., 2009.
    Outlier Detection in Axis-Parallel Subspaces of High Dimensional Data.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_array

from ..utils.param_check import check_parameter
from .baseml import BaseVisionDetector
from .registry import register_model


def _top_ref_set_indices(
    neighbor_sets: list[set[int]],
    *,
    ref_set: int,
) -> np.ndarray:
    """Compute reference indices for each training sample via SNN counts."""

    n = len(neighbor_sets)
    out = np.empty((n, ref_set), dtype=int)
    for i in range(n):
        s_i = neighbor_sets[i]
        counts = np.empty(n, dtype=int)
        for j in range(n):
            if i == j:
                counts[j] = -1
            else:
                counts[j] = len(s_i.intersection(neighbor_sets[j]))
        # Select top `ref_set` indices (deterministic ordering by count then index).
        top = np.argpartition(-counts, ref_set)[:ref_set]
        # Stable-ish deterministic tie-breaker: sort by (-count, idx).
        top = top[np.lexsort((top, -counts[top]))]
        out[i] = top
    return out


class CoreSOD:
    """Pure sklearn + NumPy implementation of SOD."""

    def __init__(
        self,
        *,
        contamination: float = 0.1,
        n_neighbors: int = 20,
        ref_set: int = 10,
        alpha: float = 0.8,
    ) -> None:
        self.contamination = float(contamination)

        if not isinstance(n_neighbors, int):
            raise TypeError(f"n_neighbors must be int, got {type(n_neighbors).__name__}")
        if not isinstance(ref_set, int):
            raise TypeError(f"ref_set must be int, got {type(ref_set).__name__}")

        check_parameter(n_neighbors, low=2, param_name="n_neighbors")
        check_parameter(ref_set, low=1, param_name="ref_set")
        if ref_set >= n_neighbors:
            raise ValueError("ref_set must be < n_neighbors")

        check_parameter(float(alpha), low=0.0, high=1.0, param_name="alpha")

        self.n_neighbors = int(n_neighbors)
        self.ref_set = int(ref_set)
        self.alpha = float(alpha)

        self._nn: NearestNeighbors | None = None
        self._X_train: np.ndarray | None = None
        self._neighbor_sets: list[set[int]] | None = None
        self._ref_inds_train: np.ndarray | None = None
        self.decision_scores_: np.ndarray | None = None

    def _score_one(self, obs: np.ndarray, ref: np.ndarray) -> float:
        means = np.mean(ref, axis=0)
        var_total = float(np.sum(np.square(ref - means)) / self.ref_set)
        var_expect = self.alpha * var_total / ref.shape[1]
        var_actual = np.var(ref, axis=0)
        var_mask = var_actual < var_expect
        rel_dim = int(np.sum(var_mask))
        if rel_dim == 0:
            return 0.0
        return float(np.sqrt(np.sum(np.square(obs - means)[var_mask]) / rel_dim))

    def fit(self, X, y=None):  # noqa: ANN001, ANN201 - sklearn/pyod-like API
        X = check_array(X, ensure_2d=True, dtype=np.float64)
        n_samples = X.shape[0]
        if n_samples <= 1:
            self._X_train = X
            self.decision_scores_ = np.zeros(n_samples, dtype=np.float64)
            return self

        k = min(self.n_neighbors, n_samples - 1)
        r = min(self.ref_set, n_samples - 1)
        if r < 1:
            self._X_train = X
            self.decision_scores_ = np.zeros(n_samples, dtype=np.float64)
            return self

        # Build kNN structure on training set.
        self._nn = NearestNeighbors(n_neighbors=k + 1)
        self._nn.fit(X)
        ind = self._nn.kneighbors(X, n_neighbors=k + 1, return_distance=False)
        ind = ind[:, 1:]  # drop self

        self._X_train = X
        self._neighbor_sets = [set(map(int, row)) for row in ind]
        self._ref_inds_train = _top_ref_set_indices(self._neighbor_sets, ref_set=r)

        scores = np.zeros(n_samples, dtype=np.float64)
        for i in range(n_samples):
            ref = X[self._ref_inds_train[i]]
            scores[i] = self._score_one(X[i], ref)
        self.decision_scores_ = scores
        return self

    def decision_function(self, X):  # noqa: ANN001, ANN201 - sklearn/pyod-like API
        if self._X_train is None or self._nn is None or self._neighbor_sets is None:
            raise RuntimeError("Detector must be fitted before calling decision_function")

        X = check_array(X, ensure_2d=True, dtype=np.float64)
        if X.shape[0] == 0:
            return np.zeros(0, dtype=np.float64)

        n_train = self._X_train.shape[0]
        if n_train <= 1:
            return np.zeros(X.shape[0], dtype=np.float64)

        k = min(self.n_neighbors, n_train)
        r = min(self.ref_set, n_train - 1)
        if r < 1:
            return np.zeros(X.shape[0], dtype=np.float64)

        ind_x = self._nn.kneighbors(X, n_neighbors=k, return_distance=False)

        scores = np.zeros(X.shape[0], dtype=np.float64)
        for i in range(X.shape[0]):
            s_x = set(map(int, ind_x[i]))
            counts = np.empty(n_train, dtype=int)
            for j in range(n_train):
                counts[j] = len(s_x.intersection(self._neighbor_sets[j]))
            top = np.argpartition(-counts, r)[:r]
            top = top[np.lexsort((top, -counts[top]))]
            ref = self._X_train[top]
            scores[i] = self._score_one(X[i], ref)
        return scores


@register_model(
    "vision_sod",
    tags=("vision", "classical", "sod", "subspace", "baseline"),
    metadata={
        "description": "Subspace Outlier Detection (subspace baseline)",
        "type": "subspace",
    },
)
class VisionSOD(BaseVisionDetector):
    """Vision-compatible SOD detector."""

    def __init__(
        self,
        *,
        feature_extractor=None,
        contamination: float = 0.1,
        n_neighbors: int = 20,
        ref_set: int = 10,
        alpha: float = 0.8,
    ) -> None:
        self._detector_kwargs = {
            "contamination": float(contamination),
            "n_neighbors": int(n_neighbors),
            "ref_set": int(ref_set),
            "alpha": float(alpha),
        }
        super().__init__(contamination=contamination, feature_extractor=feature_extractor)

    def _build_detector(self):
        return CoreSOD(**self._detector_kwargs)

    def fit(self, X: Iterable[str], y=None):
        return super().fit(X, y=y)

    def decision_function(self, X):
        return super().decision_function(X)

