# -*- coding: utf-8 -*-
"""
ABOD (Angle-Based Outlier Detection) vision integration.

ABOD uses the variance of weighted cosine values formed by (approximate)
neighbor pairs. Outliers tend to have *lower* variance, so we flip the sign to
match the convention "higher score => more anomalous".

Reference:
    Kriegel, H.-P. et al., 2008. Angle-Based Outlier Detection in High-dimensional Data.
"""

from __future__ import annotations

from itertools import combinations
from typing import Iterable

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_array

from ..utils.param_check import check_parameter
from .baseml import BaseVisionDetector
from .registry import register_model


def _weighted_cosine(curr: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    a_curr = a - curr
    b_curr = b - curr
    na2 = float(np.dot(a_curr, a_curr))
    nb2 = float(np.dot(b_curr, b_curr))
    denom = na2 * nb2
    if denom <= 0.0:
        return 0.0
    return float(np.dot(a_curr, b_curr) / denom)


def _abod_variance(curr: np.ndarray, X_train: np.ndarray, inds: np.ndarray) -> float:
    wcos = []
    for a_idx, b_idx in combinations(list(map(int, inds)), 2):
        a = X_train[a_idx]
        b = X_train[b_idx]
        # Skip degenerate angles.
        if np.array_equal(a, curr) or np.array_equal(b, curr):
            continue
        wcos.append(_weighted_cosine(curr, a, b))
    if not wcos:
        return 0.0
    return float(np.var(np.asarray(wcos, dtype=np.float64)))


class CoreABOD:
    """Native ABOD core (fast approximation by kNN)."""

    def __init__(
        self,
        *,
        contamination: float = 0.1,
        n_neighbors: int = 10,
        method: str = "fast",
    ) -> None:
        self.contamination = float(contamination)
        self.method = str(method)
        self.n_neighbors = int(n_neighbors)

        check_parameter(self.n_neighbors, low=1, param_name="n_neighbors")
        if self.method not in {"fast", "default"}:
            raise ValueError("method must be one of {'fast', 'default'}")

        self.X_train_: np.ndarray | None = None
        self.nn_: NearestNeighbors | None = None
        self.decision_scores_: np.ndarray | None = None

    def fit(self, X, y=None):  # noqa: ANN001, ANN201 - sklearn/pyod-like API
        X = check_array(X, ensure_2d=True, dtype=np.float64)
        self.X_train_ = X
        n_train = X.shape[0]
        if n_train <= 2:
            self.decision_scores_ = np.zeros(n_train, dtype=np.float64)
            return self

        if self.method == "fast":
            k = min(self.n_neighbors, n_train - 1)
            self.nn_ = NearestNeighbors(n_neighbors=k + 1).fit(X)
            ind_arr = self.nn_.kneighbors(X, n_neighbors=k + 1, return_distance=False)[:, 1:]
            scores = np.zeros(n_train, dtype=np.float64)
            for i in range(n_train):
                scores[i] = -_abod_variance(X[i], X, ind_arr[i])
            self.decision_scores_ = scores
            return self

        # Default ABOD: use all points (O(n^3)).
        all_inds = np.arange(n_train, dtype=int)
        scores = np.zeros(n_train, dtype=np.float64)
        for i in range(n_train):
            mask = all_inds != i
            scores[i] = -_abod_variance(X[i], X, all_inds[mask])
        self.decision_scores_ = scores
        return self

    def decision_function(self, X):  # noqa: ANN001, ANN201 - sklearn/pyod-like API
        if self.X_train_ is None or self.decision_scores_ is None:
            raise RuntimeError("Detector must be fitted before calling decision_function")

        X = check_array(X, ensure_2d=True, dtype=np.float64)
        n_train = self.X_train_.shape[0]
        if n_train <= 2:
            return np.zeros(X.shape[0], dtype=np.float64)

        if self.method == "fast":
            if self.nn_ is None:
                raise RuntimeError("Internal error: missing neighbor index")
            k = min(self.n_neighbors, n_train)
            ind_arr = self.nn_.kneighbors(X, n_neighbors=k, return_distance=False)
            scores = np.zeros(X.shape[0], dtype=np.float64)
            for i in range(X.shape[0]):
                scores[i] = -_abod_variance(X[i], self.X_train_, ind_arr[i])
            return scores

        # Default ABOD: compare against all training points.
        all_inds = np.arange(n_train, dtype=int)
        scores = np.zeros(X.shape[0], dtype=np.float64)
        for i in range(X.shape[0]):
            scores[i] = -_abod_variance(X[i], self.X_train_, all_inds)
        return scores


@register_model(
    "vision_abod",
    tags=("vision", "classical", "abod"),
    metadata={"description": "基于 ABOD 的视觉异常检测器 (native)"},
)
class VisionABOD(BaseVisionDetector):
    """Vision ABOD detector."""

    def __init__(
        self,
        *,
        contamination: float = 0.1,
        feature_extractor=None,
        n_neighbors: int = 10,
        method: str = "fast",
    ):
        self._detector_kwargs = dict(
            contamination=float(contamination),
            n_neighbors=int(n_neighbors),
            method=str(method),
        )
        super().__init__(contamination=contamination, feature_extractor=feature_extractor)

    def _build_detector(self):
        return CoreABOD(**self._detector_kwargs)

    def fit(self, X: Iterable[str], y=None):
        return super().fit(X, y=y)

    def decision_function(self, X):
        return super().decision_function(X)

