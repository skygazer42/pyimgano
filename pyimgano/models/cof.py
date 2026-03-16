# -*- coding: utf-8 -*-
"""
COF (Connectivity-Based Outlier Factor) detector.

COF compares the average chaining distance (ACD) of a point against the average
ACD of its k nearest neighbors.

Reference:
    Tang, J., Chen, Z., Fu, A.W.C. and Cheung, D.W., 2002.
    Enhancing effectiveness of outlier detections for low density patterns.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.spatial import distance_matrix
from sklearn.utils import check_array

from ..utils.param_check import check_parameter
from .baseml import BaseVisionDetector
from .core_feature_base import CoreFeatureDetector
from .registry import register_model


def _acd_weights(k: int) -> np.ndarray:
    # Weight for the j-th neighbor (1-indexed in the original formula).
    j = np.arange(1, k + 1, dtype=np.float64)
    return (2.0 * (k + 1.0 - j)) / ((k + 1.0) * k)


class CoreCOF:
    """Native COF implementation (fast, stores train pairwise distances)."""

    _legacy_attr_aliases = {"X_train_": "x_train_"}

    def __init__(self, *, contamination: float = 0.1, n_neighbors: int = 20) -> None:
        self.contamination = float(contamination)
        self.n_neighbors = int(n_neighbors)
        check_parameter(self.n_neighbors, low=1, param_name="n_neighbors")

        self.n_neighbors_: int | None = None
        self.x_train_: np.ndarray | None = None
        self._dist_train: np.ndarray | None = None
        self._neighbors_train: np.ndarray | None = None
        self._ac_dist_train: np.ndarray | None = None
        self.decision_scores_: np.ndarray | None = None

    def __getattr__(self, name: str):
        alias = type(self)._legacy_attr_aliases.get(name)
        if alias is not None:
            return getattr(self, alias)
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")

    def __setattr__(self, name: str, value) -> None:
        alias = type(self)._legacy_attr_aliases.get(name)
        super().__setattr__(alias or name, value)

    def _ac_dist_for_point(
        self,
        *,
        chain_order: np.ndarray,
        dist_to_query: np.ndarray,
        dist_train: np.ndarray,
        k: int,
    ) -> float:
        """Compute average chaining distance for one query point."""

        weights = _acd_weights(k)
        cost_desc = np.zeros(k, dtype=np.float64)

        prev: list[int] = []
        for j in range(k):
            idx = int(chain_order[j])

            # Minimum distance to the current chain: either to the query itself
            # (dist_to_query[idx]) or to any previous selected training point.
            best = float(dist_to_query[idx])
            if prev:
                best = min(best, float(np.min(dist_train[idx, prev])))
            cost_desc[j] = best
            prev.append(idx)

        return float(np.sum(weights * cost_desc))

    def fit(self, x, y=None):  # noqa: ANN001, ANN201 - sklearn-like API
        del y
        x = check_array(x, ensure_2d=True, dtype=np.float64)
        n_train = x.shape[0]
        self.x_train_ = x

        if n_train <= 1:
            self.n_neighbors_ = 0
            self._dist_train = None
            self._neighbors_train = None
            self._ac_dist_train = np.zeros(n_train, dtype=np.float64)
            self.decision_scores_ = np.zeros(n_train, dtype=np.float64)
            return self

        k = min(self.n_neighbors, n_train - 1)
        self.n_neighbors_ = int(k)

        dist = np.asarray(distance_matrix(x, x), dtype=np.float64)
        self._dist_train = dist

        # Neighbor ordering (excluding self).
        neighbors = np.zeros((n_train, k), dtype=int)
        for i in range(n_train):
            order = np.argsort(dist[i])
            neighbors[i] = order[1 : k + 1]
        self._neighbors_train = neighbors

        # Precompute ACD for each training point.
        ac_dist = np.zeros(n_train, dtype=np.float64)
        for i in range(n_train):
            order = neighbors[i]
            dist_to_query = dist[i]
            ac_dist[i] = self._ac_dist_for_point(
                chain_order=order,
                dist_to_query=dist_to_query,
                dist_train=dist,
                k=k,
            )
        self._ac_dist_train = ac_dist

        # COF score for training points.
        scores = np.zeros(n_train, dtype=np.float64)
        for i in range(n_train):
            denom = float(np.sum(ac_dist[neighbors[i]]))
            scores[i] = 0.0 if denom <= 0.0 else float(ac_dist[i] * k / denom)
        self.decision_scores_ = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        return self

    def decision_function(self, x):  # noqa: ANN001, ANN201 - sklearn-like API
        if (
            self.x_train_ is None
            or self.n_neighbors_ is None
            or self._ac_dist_train is None
            or self._neighbors_train is None
            or self.decision_scores_ is None
        ):
            raise RuntimeError("Detector must be fitted before calling decision_function")

        x = check_array(x, ensure_2d=True, dtype=np.float64)
        if self.n_neighbors_ == 0:
            return np.zeros(x.shape[0], dtype=np.float64)

        k = int(self.n_neighbors_)
        # Distances from query to training points.
        dist_q = np.asarray(distance_matrix(x, self.x_train_), dtype=np.float64)

        scores = np.zeros(x.shape[0], dtype=np.float64)
        for i in range(x.shape[0]):
            order = np.argsort(dist_q[i])[:k]
            acd = self._ac_dist_for_point(
                chain_order=order,
                dist_to_query=dist_q[i],
                dist_train=(
                    self._dist_train
                    if self._dist_train is not None
                    else distance_matrix(self.x_train_, self.x_train_)
                ),
                k=k,
            )
            denom = float(np.sum(self._ac_dist_train[order]))
            scores[i] = 0.0 if denom <= 0.0 else float(acd * k / denom)

        return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)


@register_model(
    "core_cof",
    tags=("classical", "core", "features", "neighbors", "cof", "density"),
    metadata={
        "description": "Core COF detector on feature matrices (native wrapper)",
        "input": "features",
        "paper": "Tang et al., PAKDD 2002",
        "year": 2002,
    },
)
class CoreCOFModel(CoreFeatureDetector):
    """Core (feature-matrix) COF detector with BaseDetector thresholding."""

    def __init__(
        self,
        *,
        contamination: float = 0.1,
        n_neighbors: int = 20,
    ) -> None:
        self._backend_kwargs = {
            "contamination": float(contamination),
            "n_neighbors": int(n_neighbors),
        }
        super().__init__(contamination=float(contamination))

    def _build_detector(self):
        return CoreCOF(**self._backend_kwargs)


@register_model(
    "vision_cof",
    tags=("vision", "classical", "neighbors", "cof"),
    metadata={
        "description": "COF - Connectivity-based outlier detector (native)",
        "paper": "Tang et al., PAKDD 2002",
        "year": 2002,
        "density_based": True,
    },
)
class VisionCOF(BaseVisionDetector):
    """Vision-compatible COF detector."""

    def __init__(
        self,
        *,
        feature_extractor=None,
        contamination: float = 0.1,
        n_neighbors: int = 20,
    ) -> None:
        self._detector_kwargs = {
            "contamination": float(contamination),
            "n_neighbors": int(n_neighbors),
        }
        super().__init__(contamination=contamination, feature_extractor=feature_extractor)

    def _build_detector(self):
        return CoreCOF(**self._detector_kwargs)

    def fit(self, x: Iterable[str], y=None):
        return super().fit(x, y=y)

    def decision_function(self, x):
        return super().decision_function(x)
