# -*- coding: utf-8 -*-
"""DBSCAN clustering baseline (modernized).

This module exposes:
- `core_dbscan`: a feature-matrix DBSCAN-inspired distance-to-core-set score
- `vision_dbscan`: a vision wrapper using feature extractors
- `dbscan_anomaly`: a backwards-compatible alias configured for structural features

DBSCAN does not support a true out-of-sample decision function. For industrial
workflows we need stable scoring on new samples, so we approximate it by:
- fitting DBSCAN on the training set
- collecting DBSCAN core samples
- scoring new samples by distance to the nearest core sample

Higher distance => more anomalous (PyImgAno convention).
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_array

from pyimgano.features.structural import StructuralFeaturesExtractor

from .baseml import BaseVisionDetector
from .core_feature_base import CoreFeatureDetector
from .registry import register_model


class CoreDBSCAN:
    """DBSCAN-inspired detector core using distance-to-core-set scoring."""

    def __init__(
        self,
        *,
        contamination: float = 0.1,
        eps: float = 0.5,
        min_samples: int = 5,
        metric: str = "minkowski",
        p: int = 2,
        n_jobs: Optional[int] = None,
        preprocessing: bool = True,
    ) -> None:
        self.contamination = float(contamination)
        self.eps = float(eps)
        self.min_samples = int(min_samples)
        self.metric = str(metric)
        self.p = int(p)
        self.n_jobs = n_jobs
        self.preprocessing = bool(preprocessing)

        self.scaler_: StandardScaler | None = None
        self.core_samples_: np.ndarray | None = None
        self.decision_scores_: np.ndarray | None = None
        self._fitted: bool = False

    def fit(self, X, y=None):  # noqa: ANN001, ANN201
        X = check_array(X, ensure_2d=True, dtype=np.float64)
        n = int(X.shape[0])
        if n == 0:
            raise ValueError("Training set cannot be empty")
        if self.eps <= 0.0:
            raise ValueError("eps must be > 0")
        if self.min_samples < 1:
            raise ValueError("min_samples must be >= 1")

        X_proc = X
        if self.preprocessing:
            self.scaler_ = StandardScaler()
            X_proc = self.scaler_.fit_transform(X_proc)
        else:
            self.scaler_ = None

        # Fit DBSCAN on training points.
        db = DBSCAN(
            eps=float(self.eps),
            min_samples=int(self.min_samples),
            metric=self.metric,
            p=self.p,
            n_jobs=self.n_jobs,
        )
        labels = db.fit_predict(X_proc)

        core_idx = getattr(db, "core_sample_indices_", None)
        if core_idx is None or len(core_idx) == 0:
            # Degenerate: no cores discovered.
            self.core_samples_ = None
            self.decision_scores_ = np.ones((n,), dtype=np.float64)
            self._fitted = True
            return self

        self.core_samples_ = np.asarray(X_proc[np.asarray(core_idx, dtype=int)], dtype=np.float64)

        # Training scores: distance to nearest core sample.
        self._fitted = True
        self.decision_scores_ = np.asarray(self.decision_function(X), dtype=np.float64)

        # If DBSCAN labeled points as noise, boost their score slightly to reflect
        # the clustering outcome while keeping a continuous value.
        noise_mask = np.asarray(labels == -1)
        if np.any(noise_mask):
            self.decision_scores_[noise_mask] = self.decision_scores_[noise_mask] + float(self.eps)

        return self

    def decision_function(self, X):  # noqa: ANN001, ANN201
        if not bool(getattr(self, "_fitted", False)):
            raise RuntimeError("Detector must be fitted before calling decision_function")

        X = check_array(X, ensure_2d=True, dtype=np.float64)
        if self.core_samples_ is None:
            return np.ones((X.shape[0],), dtype=np.float64)

        X_proc = X
        if self.preprocessing:
            if self.scaler_ is None:
                raise RuntimeError("Internal error: missing scaler")
            X_proc = self.scaler_.transform(X_proc)

        kwargs = {"metric": self.metric, "n_jobs": self.n_jobs}
        if self.metric == "minkowski":
            kwargs["p"] = int(self.p)
        dists = pairwise_distances(X_proc, self.core_samples_, **kwargs)
        return np.min(dists, axis=1).astype(np.float64, copy=False).reshape(-1)


@register_model(
    "core_dbscan",
    tags=("classical", "core", "features", "clustering", "dbscan", "density"),
    metadata={
        "description": "Core DBSCAN-inspired distance-to-core-set detector on feature matrices",
        "input": "features",
        "paper": "A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise",
        "year": 1996,
    },
)
class CoreDBSCANModel(CoreFeatureDetector):
    """Core (feature-matrix) DBSCAN-inspired detector with BaseDetector thresholding."""

    def __init__(
        self,
        *,
        contamination: float = 0.1,
        eps: float = 0.5,
        min_samples: int = 5,
        metric: str = "minkowski",
        p: int = 2,
        n_jobs: Optional[int] = None,
        preprocessing: bool = True,
    ) -> None:
        self._backend_kwargs = {
            "contamination": float(contamination),
            "eps": float(eps),
            "min_samples": int(min_samples),
            "metric": str(metric),
            "p": int(p),
            "n_jobs": n_jobs,
            "preprocessing": bool(preprocessing),
        }
        super().__init__(contamination=float(contamination))

    def _build_detector(self):
        return CoreDBSCAN(**self._backend_kwargs)


@register_model(
    "vision_dbscan",
    tags=("vision", "classical", "clustering", "dbscan", "density"),
    metadata={
        "description": "Vision wrapper for DBSCAN-inspired distance-to-core-set baseline",
        "paper": "A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise",
        "year": 1996,
    },
)
class VisionDBSCAN(BaseVisionDetector):
    def __init__(
        self,
        *,
        feature_extractor=None,
        contamination: float = 0.1,
        eps: float = 0.5,
        min_samples: int = 5,
        metric: str = "minkowski",
        p: int = 2,
        n_jobs: Optional[int] = None,
        preprocessing: bool = True,
    ) -> None:
        self._detector_kwargs = {
            "contamination": float(contamination),
            "eps": float(eps),
            "min_samples": int(min_samples),
            "metric": str(metric),
            "p": int(p),
            "n_jobs": n_jobs,
            "preprocessing": bool(preprocessing),
        }
        super().__init__(contamination=contamination, feature_extractor=feature_extractor)

    def _build_detector(self):
        return CoreDBSCAN(**self._detector_kwargs)

    def fit(self, X: Iterable[str], y=None):
        return super().fit(X, y=y)

    def decision_function(self, X):
        return super().decision_function(X)


@register_model(
    "dbscan_anomaly",
    tags=("vision", "classical", "clustering", "dbscan", "structural"),
    metadata={
        "description": "Structural-features DBSCAN-inspired anomaly baseline (modernized)",
        "legacy_name": True,
        "paper": "A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise",
        "year": 1996,
    },
    overwrite=True,
)
class DBSCANAnomaly(VisionDBSCAN):
    """Backwards-compatible alias for structural DBSCAN anomaly baseline."""

    def __init__(
        self,
        *,
        feature_extractor=None,
        contamination: float = 0.05,
        eps: float = 0.35,
        min_samples: int = 5,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> None:
        # DBSCAN itself is deterministic; `random_state` is accepted for API
        # symmetry with other baselines and ignored.
        _ = random_state

        if feature_extractor is None:
            feature_extractor = StructuralFeaturesExtractor(max_size=512, error_mode="zeros")

        super().__init__(
            feature_extractor=feature_extractor,
            contamination=float(contamination),
            eps=float(eps),
            min_samples=int(min_samples),
            **dict(kwargs),
        )
