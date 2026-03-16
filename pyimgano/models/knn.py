# -*- coding: utf-8 -*-
"""
KNN (K-Nearest Neighbors) outlier detector.

KNN is a classic distance-based outlier detection method: points that are far
from their k nearest neighbors are considered outliers.

Reference:
    Ramaswamy, S., Rastogi, R. and Shim, K., 2000.
    Efficient algorithms for mining outliers from large data sets.
    ACM SIGMOD Record.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_array

from .baseml import BaseVisionDetector
from .core_feature_base import CoreFeatureDetector
from .registry import register_model


class CoreKNN:
    """Pure sklearn implementation of a sklearn-style KNN outlier detector."""

    _legacy_attr_aliases = {"_X_train": "_x_train"}

    def __init__(
        self,
        *,
        contamination: float = 0.1,
        n_neighbors: int = 5,
        method: str = "largest",
        algorithm: str = "auto",
        leaf_size: int = 30,
        metric: str = "minkowski",
        p: int = 2,
        metric_params=None,
        n_jobs: int = 1,
        radius: float | None = None,  # API compat (unused)
        **nn_kwargs,
    ) -> None:
        self.contamination = float(contamination)
        self.n_neighbors = int(n_neighbors)
        self.method = str(method)
        self.algorithm = str(algorithm)
        self.leaf_size = int(leaf_size)
        self.metric = str(metric)
        self.p = int(p)
        self.metric_params = metric_params
        self.n_jobs = int(n_jobs)
        self.radius = radius
        self._nn_kwargs = dict(nn_kwargs)

        # Avoid ambiguous overrides: these must be passed explicitly.
        reserved = {
            "n_neighbors",
            "radius",
            "algorithm",
            "leaf_size",
            "metric",
            "p",
            "metric_params",
            "n_jobs",
        }
        bad = reserved.intersection(self._nn_kwargs)
        if bad:
            bad_s = ", ".join(sorted(bad))
            raise TypeError(f"Pass {bad_s} explicitly (not via **kwargs)")

        self._nn: NearestNeighbors | None = None
        self._k_effective: int | None = None
        self._x_train: np.ndarray | None = None
        self.decision_scores_: np.ndarray | None = None

    def __getattr__(self, name: str):
        alias = type(self)._legacy_attr_aliases.get(name)
        if alias is not None:
            return getattr(self, alias)
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")

    def __setattr__(self, name: str, value) -> None:
        alias = type(self)._legacy_attr_aliases.get(name)
        super().__setattr__(alias or name, value)

    def _aggregate(self, distances: np.ndarray) -> np.ndarray:
        if self.method == "largest":
            return distances.max(axis=1)
        if self.method == "mean":
            return distances.mean(axis=1)
        if self.method == "median":
            return np.median(distances, axis=1)
        raise ValueError("method must be one of {'largest', 'mean', 'median'}")

    def fit(self, x, y=None):  # noqa: ANN001, ANN201 - sklearn-like API
        del y
        x = check_array(x, ensure_2d=True, dtype=np.float64)
        self._x_train = x

        n_train = x.shape[0]
        if n_train <= 1:
            # Not enough points to define neighbors; treat everything as equally normal.
            self._k_effective = 0
            self._nn = None
            self.decision_scores_ = np.zeros(n_train, dtype=np.float64)
            return self

        if self.n_neighbors < 1:
            raise ValueError("n_neighbors must be >= 1")

        k = min(self.n_neighbors, n_train - 1)
        self._k_effective = int(k)

        self._nn = NearestNeighbors(
            n_neighbors=k + 1,  # include self; dropped when scoring training points
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            metric=self.metric,
            p=self.p,
            metric_params=self.metric_params,
            n_jobs=self.n_jobs,
            **self._nn_kwargs,
        )
        self._nn.fit(x)

        distances, _ = self._nn.kneighbors(x, n_neighbors=k + 1, return_distance=True)
        distances = distances[:, 1:]  # drop self-distance
        self.decision_scores_ = self._aggregate(distances).astype(np.float64)
        return self

    def decision_function(self, x):  # noqa: ANN001, ANN201 - sklearn-like API
        if self._x_train is None:
            raise RuntimeError("Detector must be fitted before calling decision_function")

        x = check_array(x, ensure_2d=True, dtype=np.float64)
        if self._k_effective is None:
            raise RuntimeError("Internal error: missing k")
        if self._k_effective == 0:
            return np.zeros(x.shape[0], dtype=np.float64)
        if self._nn is None:
            raise RuntimeError("Internal error: missing neighbor index")

        distances, _ = self._nn.kneighbors(x, n_neighbors=self._k_effective, return_distance=True)
        return self._aggregate(distances).astype(np.float64).ravel()


@register_model(
    "core_knn",
    tags=("classical", "core", "features", "neighbors", "knn"),
    metadata={
        "description": "Core KNN outlier detector on feature matrices (native wrapper)",
        "input": "features",
    },
)
class CoreKNNModel(CoreFeatureDetector):
    """Core (feature-matrix) KNN detector with BaseDetector thresholding."""

    def __init__(
        self,
        *,
        contamination: float = 0.1,
        n_neighbors: int = 5,
        method: str = "largest",
        algorithm: str = "auto",
        leaf_size: int = 30,
        metric: str = "minkowski",
        p: int = 2,
        metric_params=None,
        n_jobs: int = 1,
        radius: float | None = None,
        **nn_kwargs,
    ) -> None:
        self._backend_kwargs = dict(
            contamination=float(contamination),
            n_neighbors=int(n_neighbors),
            method=str(method),
            algorithm=str(algorithm),
            leaf_size=int(leaf_size),
            metric=str(metric),
            p=int(p),
            metric_params=metric_params,
            n_jobs=int(n_jobs),
            radius=radius,
            **dict(nn_kwargs),
        )
        super().__init__(contamination=float(contamination))

    def _build_detector(self):
        return CoreKNN(**self._backend_kwargs)


@register_model(
    "vision_knn",
    tags=("vision", "classical", "neighbors", "knn"),
    metadata={
        "description": "Vision wrapper for KNN outlier detector",
        "paper": "SIGMOD 2000",
        "year": 2000,
        "simple": True,
        "interpretable": True,
    },
)
class VisionKNN(BaseVisionDetector):
    """Vision-compatible KNN detector for anomaly detection."""

    def __init__(
        self,
        *,
        feature_extractor=None,
        contamination: float = 0.1,
        n_neighbors: int = 5,
        method: str = "largest",
        radius: float = 1.0,
        algorithm: str = "auto",
        leaf_size: int = 30,
        metric: str = "minkowski",
        p: int = 2,
        metric_params=None,
        n_jobs: int = 1,
        **kwargs,
    ):
        self.detector_kwargs = dict(
            contamination=contamination,
            n_neighbors=n_neighbors,
            method=method,
            radius=radius,
            algorithm=algorithm,
            leaf_size=leaf_size,
            metric=metric,
            p=p,
            metric_params=metric_params,
            n_jobs=n_jobs,
            **kwargs,
        )
        super().__init__(contamination=contamination, feature_extractor=feature_extractor)

    def _build_detector(self):
        return CoreKNN(**self.detector_kwargs)

    def fit(self, x: Iterable[str], y=None):
        return super().fit(x, y=y)

    def decision_function(self, x):
        return super().decision_function(x)
