# -*- coding: utf-8 -*-
"""Random Projection + kNN distance baseline.

Motivation:
- High-dimensional embeddings can make exact kNN expensive/noisy.
- A simple random projection often preserves neighborhood structure well enough.

Pipeline:
  X -> random projection -> kNN outlier score (distance aggregation)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.utils.validation import check_array

from ..utils.fitted import require_fitted
from .baseml import BaseVisionDetector
from .core_feature_base import CoreFeatureDetector
from .knn import CoreKNN
from .registry import register_model


class _RPkNNBackend:
    def __init__(
        self,
        *,
        contamination: float,
        n_components: int | float,
        random_state: Optional[int],
        # KNN params (subset)
        n_neighbors: int,
        method: str,
        metric: str,
        p: int,
        n_jobs: int | None,
        eps: float,
    ) -> None:
        self.contamination = float(contamination)
        self.n_components = n_components
        self.random_state = random_state
        self.n_neighbors = int(n_neighbors)
        self.method = str(method)
        self.metric = str(metric)
        self.p = int(p)
        self.n_jobs = n_jobs
        self.eps = float(eps)

        self.proj_ = None
        self.backend_ = None
        self.decision_scores_ = None

    def _resolve_k(self, d: int) -> int:
        nc = self.n_components
        if isinstance(nc, float):
            if not (0.0 < nc <= 1.0):
                raise ValueError("n_components as float must be in (0,1]")
            k = int(np.ceil(float(nc) * float(d)))
        else:
            k = int(nc)
        k = max(1, min(k, int(d)))
        return k

    def _make_proj(self, d: int) -> np.ndarray:
        k = self._resolve_k(d)
        rng = np.random.default_rng(None if self.random_state is None else int(self.random_state))
        projection = rng.normal(0.0, 1.0, size=(d, k)).astype(np.float64)
        # Scale to preserve variance roughly.
        projection = projection / max(np.sqrt(float(k)), float(self.eps))
        return projection

    def _project(self, x: np.ndarray) -> np.ndarray:
        require_fitted(self, ["proj_"])
        projection = np.asarray(self.proj_, dtype=np.float64)  # type: ignore[arg-type]
        return np.asarray(x @ projection, dtype=np.float64)

    def fit(self, x, y=None):  # noqa: ANN001, ANN201 - sklearn-like
        del y
        x_arr = check_array(x, ensure_2d=True, dtype=np.float64)
        d = int(x_arr.shape[1])
        self.proj_ = self._make_proj(d)

        z = x_arr @ self.proj_

        knn = CoreKNN(
            contamination=float(self.contamination),
            n_neighbors=int(self.n_neighbors),
            method=str(self.method),
            metric=str(self.metric),
            p=int(self.p),
            n_jobs=(1 if self.n_jobs is None else int(self.n_jobs)),
        )
        knn.fit(z)
        self.backend_ = knn
        self.decision_scores_ = np.asarray(knn.decision_scores_, dtype=np.float64).reshape(-1)
        return self

    def decision_function(self, x):  # noqa: ANN001, ANN201 - sklearn-like
        require_fitted(self, ["backend_", "proj_"])
        x_arr = check_array(x, ensure_2d=True, dtype=np.float64)
        z = self._project(x_arr)
        knn: CoreKNN = self.backend_  # type: ignore[assignment]
        return np.asarray(knn.decision_function(z), dtype=np.float64).reshape(-1)


@register_model(
    "core_random_projection_knn",
    tags=("classical", "core", "features", "neighbors", "knn", "projection"),
    metadata={
        "description": "Random projection + kNN distance outlier score (native wrapper)",
        "type": "neighbors",
    },
)
class CoreRandomProjectionKNN(CoreFeatureDetector):
    def __init__(
        self,
        *,
        contamination: float = 0.1,
        n_components: int | float = 0.5,
        random_state: Optional[int] = 0,
        n_neighbors: int = 5,
        method: str = "largest",
        metric: str = "minkowski",
        p: int = 2,
        n_jobs: int | None = None,
        eps: float = 1e-12,
    ) -> None:
        self._backend_kwargs = {
            "contamination": float(contamination),
            "n_components": n_components,
            "random_state": random_state,
            "n_neighbors": int(n_neighbors),
            "method": str(method),
            "metric": str(metric),
            "p": int(p),
            "n_jobs": n_jobs,
            "eps": float(eps),
        }
        super().__init__(contamination=contamination)

    def _build_detector(self):  # noqa: ANN201
        return _RPkNNBackend(**self._backend_kwargs)


@register_model(
    "vision_random_projection_knn",
    tags=("vision", "classical", "neighbors", "knn", "projection"),
    metadata={"description": "Vision wrapper for random projection + kNN detector"},
)
class VisionRandomProjectionKNN(BaseVisionDetector):
    def __init__(
        self,
        *,
        feature_extractor=None,
        contamination: float = 0.1,
        n_components: int | float = 0.5,
        random_state: Optional[int] = 0,
        n_neighbors: int = 5,
        method: str = "largest",
        metric: str = "minkowski",
        p: int = 2,
        n_jobs: int | None = None,
        eps: float = 1e-12,
    ) -> None:
        self._detector_kwargs = {
            "contamination": float(contamination),
            "n_components": n_components,
            "random_state": random_state,
            "n_neighbors": int(n_neighbors),
            "method": str(method),
            "metric": str(metric),
            "p": int(p),
            "n_jobs": n_jobs,
            "eps": float(eps),
        }
        super().__init__(contamination=contamination, feature_extractor=feature_extractor)

    def _build_detector(self):
        # Use the core backend directly (vision wrapper calls detector.fit on extracted features).
        return _RPkNNBackend(**self._detector_kwargs)
