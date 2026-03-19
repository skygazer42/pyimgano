# -*- coding: utf-8 -*-
"""kNN epsilon-graph degree detector.

We first choose a radius `r` based on kNN distances on the training set, then
define the (epsilon-graph) degree of a point as the number of training points
within that radius. Low degree indicates an outlier in sparse regions.

Score = 1 / (degree + 1)
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_array

from ..utils.fitted import require_fitted
from ._legacy_x import MISSING, resolve_legacy_x_keyword
from .base_detector import BaseDetector
from .baseml import BaseVisionDetector
from .registry import register_model

logger = logging.getLogger(__name__)


@register_model(
    "core_knn_degree",
    tags=("classical", "core", "features", "neighbors", "graph", "density"),
    metadata={"description": "kNN epsilon-graph degree (radius chosen from kNN distances)"},
)
class CoreKNNGraphDegree(BaseDetector):
    def __init__(
        self,
        *,
        contamination: float = 0.1,
        n_neighbors: int = 10,
        radius_quantile: float = 0.5,
        metric: str = "minkowski",
        p: int = 2,
        n_jobs: int | None = None,
        zero_eps: float = 1e-12,
    ) -> None:
        super().__init__(contamination=float(contamination))
        self.n_neighbors = int(n_neighbors)
        self.radius_quantile = float(radius_quantile)
        self.metric = str(metric)
        self.p = int(p)
        self.n_jobs = n_jobs
        self.zero_eps = float(zero_eps)

    def fit(self, x: object = MISSING, y=None, **kwargs: object):  # noqa: ANN001, ANN201
        x_arr = check_array(
            resolve_legacy_x_keyword(x, kwargs, method_name="fit"),
            ensure_2d=True,
            dtype=np.float64,
        )
        self._set_n_classes(y)

        n = int(x_arr.shape[0])
        k = int(self.n_neighbors)
        if k <= 0:
            raise ValueError("n_neighbors must be > 0")
        if n <= k:
            raise ValueError(f"Need n_samples > n_neighbors, got n={n} k={k}")
        q = float(self.radius_quantile)
        if not (0.0 < q <= 1.0):
            raise ValueError(f"radius_quantile must be in (0,1], got {q}")

        nn = NearestNeighbors(
            n_neighbors=k + 1,
            metric=self.metric,
            p=self.p,
            n_jobs=self.n_jobs,
        )
        nn.fit(x_arr)
        distances, _idx = nn.kneighbors(x_arr, n_neighbors=k + 1, return_distance=True)
        kth = np.asarray(distances[:, -1], dtype=np.float64)
        radius = float(np.quantile(kth, q))

        self._nn = nn
        self.radius_ = radius

        self.decision_scores_ = np.asarray(self.decision_function(x_arr), dtype=np.float64)
        self._process_decision_scores()
        return self

    def decision_function(self, x: object = MISSING, **kwargs: object):  # noqa: ANN001, ANN201
        require_fitted(self, ["_nn", "radius_"])
        nn: NearestNeighbors = self._nn  # type: ignore[assignment]
        radius = float(self.radius_)  # type: ignore[arg-type]

        x_arr = check_array(
            resolve_legacy_x_keyword(x, kwargs, method_name="decision_function"),
            ensure_2d=True,
            dtype=np.float64,
        )
        if x_arr.shape[0] == 0:
            return np.zeros((0,), dtype=np.float64)

        dists, _ = nn.radius_neighbors(x_arr, radius=radius, return_distance=True)
        degrees = np.empty((x_arr.shape[0],), dtype=np.float64)
        for i in range(x_arr.shape[0]):
            # Exclude exact matches (distance ~ 0) to avoid counting self when present.
            di = np.asarray(dists[i], dtype=np.float64)
            degrees[i] = float(np.sum(di > float(self.zero_eps)))

        scores = 1.0 / (degrees + 1.0)
        return np.asarray(scores, dtype=np.float64).reshape(-1)


@register_model(
    "vision_knn_degree",
    tags=("vision", "classical", "neighbors", "graph", "density"),
    metadata={"description": "Vision kNN epsilon-graph degree"},
)
class VisionKNNGraphDegree(BaseVisionDetector):
    def __init__(
        self,
        *,
        feature_extractor=None,
        contamination: float = 0.1,
        n_neighbors: int = 10,
        radius_quantile: float = 0.5,
        metric: str = "minkowski",
        p: int = 2,
        n_jobs: int | None = None,
        zero_eps: float = 1e-12,
    ) -> None:
        self._detector_kwargs = {
            "contamination": float(contamination),
            "n_neighbors": int(n_neighbors),
            "radius_quantile": float(radius_quantile),
            "metric": str(metric),
            "p": int(p),
            "n_jobs": n_jobs,
            "zero_eps": float(zero_eps),
        }
        logger.debug("Initializing VisionKNNGraphDegree with kwargs=%s", self._detector_kwargs)
        super().__init__(contamination=contamination, feature_extractor=feature_extractor)

    def _build_detector(self):
        return CoreKNNGraphDegree(**self._detector_kwargs)
