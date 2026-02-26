# -*- coding: utf-8 -*-
"""Sampling-based distance outlier detector (SP).

Rapid distance-based outlier detection via sampling:
1) sample a subset of the training data once
2) anomaly score is the distance to the closest sampled point

Reference
---------
Sugiyama, M. and Borgwardt, K.M., 2013. Rapid Distance-Based Outlier Detection
via Sampling. NeurIPS (NIPS 2013).

Notes
-----
This is a native PyImgAno implementation (no `pyod` dependency).
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import pairwise_distances
from sklearn.utils import check_array, check_random_state

from .baseml import BaseVisionDetector
from .registry import register_model


class CoreSampling:
    """Sampling-based detector core operating on feature vectors."""

    def __init__(
        self,
        *,
        contamination: float = 0.1,
        subset_size: float | int = 20,
        metric: str = "minkowski",
        metric_params: Optional[dict[str, Any]] = None,
        random_state: Optional[int] = None,
    ) -> None:
        self.contamination = float(contamination)
        self.subset_size = subset_size
        self.metric = str(metric)
        self.metric_params = dict(metric_params) if metric_params is not None else None
        self.random_state = random_state

        self._rng = check_random_state(random_state)
        self.subset_: NDArray[np.float64] | None = None
        self.decision_scores_: NDArray[np.float64] | None = None

    def _resolve_subset_size(self, n_samples: int) -> int:
        if isinstance(self.subset_size, bool):
            raise TypeError("subset_size must be int or float, not bool")

        if isinstance(self.subset_size, int):
            if not (0 < self.subset_size <= n_samples):
                raise ValueError(
                    f"subset_size={self.subset_size} must be in (0, n_samples={n_samples}]"
                )
            return int(self.subset_size)

        subset_f = float(self.subset_size)
        if not (0.0 < subset_f <= 1.0):
            raise ValueError("subset_size as float must be in (0.0, 1.0]")
        return max(1, int(subset_f * n_samples))

    def fit(self, X, y=None):  # noqa: ANN001, ANN201 - sklearn/pyod-like API
        X = check_array(X, ensure_2d=True, dtype=np.float64)
        n_samples = int(X.shape[0])
        if n_samples == 0:
            raise ValueError("Training set cannot be empty")

        subset_size = self._resolve_subset_size(n_samples)
        idx = self._rng.choice(n_samples, size=subset_size, replace=False)
        self.subset_ = X[idx, :]

        self.decision_scores_ = np.asarray(self.decision_function(X), dtype=np.float64)
        return self

    def decision_function(self, X):  # noqa: ANN001, ANN201 - sklearn/pyod-like API
        if self.subset_ is None:
            raise RuntimeError("Detector must be fitted before calling decision_function")

        X = check_array(X, ensure_2d=True, dtype=np.float64)

        distances = pairwise_distances(
            X,
            self.subset_,
            metric=self.metric,
            **(self.metric_params or {}),
        )
        return np.min(distances, axis=1).astype(np.float64, copy=False).reshape(-1)


@register_model(
    "vision_sampling",
    tags=("vision", "classical", "sampling", "distance"),
    metadata={"description": "Sampling-based distance outlier detector (native)"},
)
class VisionSampling(BaseVisionDetector):
    """Vision-friendly Sampling wrapper around the core detector."""

    def __init__(
        self,
        *,
        feature_extractor=None,
        contamination: float = 0.1,
        subset_size: float | int = 20,
        metric: str = "minkowski",
        metric_params: Optional[dict[str, Any]] = None,
        random_state: Optional[int] = None,
    ) -> None:
        self._detector_kwargs = dict(
            contamination=float(contamination),
            subset_size=subset_size,
            metric=str(metric),
            metric_params=metric_params,
            random_state=random_state,
        )
        super().__init__(contamination=contamination, feature_extractor=feature_extractor)

    def _build_detector(self):
        return CoreSampling(**self._detector_kwargs)

    def fit(self, X: Iterable[str], y=None):
        return super().fit(X, y=y)

    def decision_function(self, X):
        return super().decision_function(X)

