# -*- coding: utf-8 -*-
"""LID (Local Intrinsic Dimensionality) outlier score.

LID is a simple kNN-distance based statistic often used for OOD/anomaly scoring.
We treat higher LID as more anomalous (industrial baseline).
"""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_array

from pyimgano.utils.fitted import require_fitted

from .base_detector import BaseDetector
from .baseml import BaseVisionDetector
from .registry import register_model


def _lid_from_knn_distances(dist: np.ndarray, *, eps: float) -> np.ndarray:
    d = np.asarray(dist, dtype=np.float64)
    if d.ndim != 2:
        raise ValueError(f"Expected knn distance matrix shape (N,k), got {d.shape}")
    if d.shape[1] < 1:
        return np.zeros((d.shape[0],), dtype=np.float64)

    # r_k is the farthest neighbor distance among the k neighbors.
    rk = np.asarray(d[:, [-1]], dtype=np.float64)
    rk = np.maximum(rk, float(eps))
    ratio = np.maximum(d, float(eps)) / rk
    # ratio in (0,1]; log <= 0
    lid = -np.mean(np.log(ratio), axis=1)
    return np.asarray(lid, dtype=np.float64).reshape(-1)


@register_model(
    "core_lid",
    tags=("classical", "core", "features", "neighbors", "lid"),
    metadata={
        "description": "Local Intrinsic Dimensionality (LID) kNN-distance outlier score",
        "type": "neighbors",
    },
)
class CoreLID(BaseDetector):
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

    def fit(self, x, y=None):  # noqa: ANN001, ANN201
        x_arr = check_array(x, ensure_2d=True, dtype=np.float64)
        self._set_n_classes(y)

        n = int(x_arr.shape[0])
        k = int(self.n_neighbors)
        if k <= 0:
            raise ValueError(f"n_neighbors must be > 0, got {self.n_neighbors}")
        if n <= 1:
            self._X_train = x_arr
            self._nn = NearestNeighbors(n_neighbors=1, metric=self.metric).fit(x_arr)
            self.decision_scores_ = np.zeros((n,), dtype=np.float64)
            self._process_decision_scores()
            return self
        if n <= k:
            # For training we need to exclude self; require at least k+1.
            k = n - 1

        nn = NearestNeighbors(
            n_neighbors=k + 1,
            metric=self.metric,
            p=self.p,
            n_jobs=self.n_jobs,
        )
        nn.fit(x_arr)
        dist, _idx = nn.kneighbors(x_arr, n_neighbors=k + 1, return_distance=True)
        dist = np.asarray(dist[:, 1:], dtype=np.float64)  # drop self

        scores = _lid_from_knn_distances(dist, eps=float(self.eps))
        self._X_train = x_arr
        self._nn = nn
        self.decision_scores_ = scores
        self._process_decision_scores()
        return self

    def decision_function(self, x):  # noqa: ANN001, ANN201
        require_fitted(self, ["_nn", "_X_train"])
        x_arr = check_array(x, ensure_2d=True, dtype=np.float64)
        nn: NearestNeighbors = self._nn  # type: ignore[assignment]

        # For new points, use k neighbors from training set.
        k = int(self.n_neighbors)
        n_train = int(getattr(self, "_X_train").shape[0])  # type: ignore[union-attr]
        if n_train <= 1:
            return np.zeros((x_arr.shape[0],), dtype=np.float64)

        k_eff = min(k, n_train)
        if k_eff <= 0:
            return np.zeros((x_arr.shape[0],), dtype=np.float64)

        dist, _idx = nn.kneighbors(x_arr, n_neighbors=k_eff, return_distance=True)
        return _lid_from_knn_distances(dist, eps=float(self.eps))


@register_model(
    "vision_lid",
    tags=("vision", "classical", "neighbors", "lid"),
    metadata={"description": "Vision wrapper for LID kNN-distance outlier score"},
)
class VisionLID(BaseVisionDetector):
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
        return CoreLID(**self._detector_kwargs)
