# -*- coding: utf-8 -*-
"""KMeans clustering baseline (modernized).

This module exposes:
- `core_kmeans`: a feature-matrix KMeans distance-to-centroid baseline
- `vision_kmeans`: a vision wrapper using feature extractors
- `kmeans_anomaly`: a backwards-compatible alias configured for structural features

The previous `kmeans_anomaly` implementation was a script-like detector with
ad-hoc feature extraction and custom thresholding logic.

We now rebuild it on the stable `BaseDetector` / `BaseVisionDetector` contract.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.utils import check_array

from pyimgano.features.structural import StructuralFeaturesExtractor

from .baseml import BaseVisionDetector
from .core_feature_base import CoreFeatureDetector
from .registry import register_model


class CoreKMeans:
    """Sklearn-backed KMeans distance-to-centroid baseline."""

    def __init__(
        self,
        *,
        contamination: float = 0.1,
        n_clusters: int = 8,
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
        algorithm: str = "lloyd",
        **kwargs,
    ) -> None:
        self.contamination = float(contamination)
        self.n_clusters = int(n_clusters)
        self.n_init = int(n_init)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = random_state
        self.algorithm = str(algorithm)
        self._kwargs = dict(kwargs)

        self.kmeans_: KMeans | None = None
        self.decision_scores_: np.ndarray | None = None

    def fit(self, X, y=None):  # noqa: ANN001, ANN201
        X = check_array(X, ensure_2d=True, dtype=np.float64)
        n = int(X.shape[0])
        if n == 0:
            raise ValueError("Training set cannot be empty")

        k = int(self.n_clusters)
        if k < 1:
            raise ValueError("n_clusters must be >= 1")
        if k > n:
            raise ValueError(f"n_clusters must be <= n_samples, got k={k} n={n}")

        km = KMeans(
            n_clusters=k,
            n_init=self.n_init,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
            algorithm=self.algorithm,
            **self._kwargs,
        )
        km.fit(X)
        self.kmeans_ = km

        self.decision_scores_ = np.asarray(self.decision_function(X), dtype=np.float64)
        return self

    def decision_function(self, X):  # noqa: ANN001, ANN201
        if self.kmeans_ is None:
            raise RuntimeError("Detector must be fitted before calling decision_function")

        X = check_array(X, ensure_2d=True, dtype=np.float64)
        centers = np.asarray(self.kmeans_.cluster_centers_, dtype=np.float64)
        dists = pairwise_distances(X, centers, metric="euclidean")
        return np.min(dists, axis=1).astype(np.float64, copy=False).reshape(-1)


@register_model(
    "core_kmeans",
    tags=("classical", "core", "features", "clustering", "kmeans"),
    metadata={
        "description": "Core KMeans distance-to-centroid baseline on feature matrices",
        "input": "features",
    },
)
class CoreKMeansModel(CoreFeatureDetector):
    """Core (feature-matrix) KMeans detector with BaseDetector thresholding."""

    def __init__(
        self,
        *,
        contamination: float = 0.1,
        n_clusters: int = 8,
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
        algorithm: str = "lloyd",
        **kwargs,
    ) -> None:
        self._backend_kwargs = dict(
            contamination=float(contamination),
            n_clusters=int(n_clusters),
            n_init=int(n_init),
            max_iter=int(max_iter),
            tol=float(tol),
            random_state=random_state,
            algorithm=str(algorithm),
            **dict(kwargs),
        )
        super().__init__(contamination=float(contamination))

    def _build_detector(self):
        return CoreKMeans(**self._backend_kwargs)


@register_model(
    "vision_kmeans",
    tags=("vision", "classical", "clustering", "kmeans"),
    metadata={"description": "Vision wrapper for KMeans distance-to-centroid baseline"},
)
class VisionKMeans(BaseVisionDetector):
    def __init__(
        self,
        *,
        feature_extractor=None,
        contamination: float = 0.1,
        n_clusters: int = 8,
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
        algorithm: str = "lloyd",
        **kwargs,
    ) -> None:
        self._detector_kwargs = dict(
            contamination=float(contamination),
            n_clusters=int(n_clusters),
            n_init=int(n_init),
            max_iter=int(max_iter),
            tol=float(tol),
            random_state=random_state,
            algorithm=str(algorithm),
            **dict(kwargs),
        )
        super().__init__(contamination=contamination, feature_extractor=feature_extractor)

    def _build_detector(self):
        return CoreKMeans(**self._detector_kwargs)

    def fit(self, X: Iterable[str], y=None):
        return super().fit(X, y=y)

    def decision_function(self, X):
        return super().decision_function(X)


@register_model(
    "kmeans_anomaly",
    tags=("vision", "classical", "clustering", "kmeans", "structural"),
    metadata={
        "description": "Structural-features KMeans anomaly baseline (modernized)",
        "legacy_name": True,
    },
    overwrite=True,
)
class KMeansAnomaly(VisionKMeans):
    """Backwards-compatible alias for structural KMeans anomaly baseline."""

    def __init__(
        self,
        *,
        feature_extractor=None,
        contamination: float = 0.05,
        n_clusters: int = 10,
        random_state: Optional[int] = 42,
        **kwargs,
    ) -> None:
        if feature_extractor is None:
            feature_extractor = StructuralFeaturesExtractor(max_size=512, error_mode="zeros")
        super().__init__(
            feature_extractor=feature_extractor,
            contamination=float(contamination),
            n_clusters=int(n_clusters),
            random_state=random_state,
            **dict(kwargs),
        )
