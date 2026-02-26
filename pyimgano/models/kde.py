# -*- coding: utf-8 -*-
"""
KDE (Kernel Density Estimation) detector.

This detector fits a density model and uses negative log-likelihood as an
outlier score (higher = more anomalous).
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.utils import check_array

from .baseml import BaseVisionDetector
from .registry import register_model


class CoreKDE:
    """Sklearn-backed KDE anomaly detector."""

    def __init__(
        self,
        *,
        contamination: float = 0.1,
        bandwidth: float = 1.0,
        algorithm: str = "auto",
        leaf_size: int = 30,
        metric: str = "minkowski",
        metric_params: dict[str, Any] | None = None,
        kernel: str = "gaussian",
        atol: float = 0.0,
        rtol: float = 0.0,
        breadth_first: bool = True,
        **kde_kwargs,
    ) -> None:
        self.contamination = float(contamination)
        self.bandwidth = float(bandwidth)
        self.algorithm = str(algorithm)
        self.leaf_size = int(leaf_size)
        self.metric = str(metric)
        self.metric_params = metric_params
        self.kernel = str(kernel)
        self.atol = float(atol)
        self.rtol = float(rtol)
        self.breadth_first = bool(breadth_first)
        self._kde_kwargs = dict(kde_kwargs)

        self.kde_: KernelDensity | None = None
        self.decision_scores_: np.ndarray | None = None

    def fit(self, X, y=None):  # noqa: ANN001, ANN201
        X = check_array(X, ensure_2d=True, dtype=np.float64)
        self.kde_ = KernelDensity(
            bandwidth=self.bandwidth,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            metric=self.metric,
            metric_params=self.metric_params,
            kernel=self.kernel,
            atol=self.atol,
            rtol=self.rtol,
            breadth_first=self.breadth_first,
            **self._kde_kwargs,
        )
        self.kde_.fit(X)
        self.decision_scores_ = self.decision_function(X)
        return self

    def decision_function(self, X):  # noqa: ANN001, ANN201
        if self.kde_ is None:
            raise RuntimeError("Detector must be fitted before calling decision_function")
        X = check_array(X, ensure_2d=True, dtype=np.float64)
        log_density = self.kde_.score_samples(X)
        return (-log_density).astype(np.float64).ravel()


@register_model(
    "vision_kde",
    tags=("vision", "classical", "kde", "density", "baseline"),
    metadata={
        "description": "Kernel Density Estimation detector (density baseline)",
        "type": "density",
    },
)
class VisionKDE(BaseVisionDetector):
    """Vision-compatible KDE detector."""

    def __init__(
        self,
        *,
        feature_extractor=None,
        contamination: float = 0.1,
        bandwidth: float = 1.0,
        algorithm: str = "auto",
        leaf_size: int = 30,
        metric: str = "minkowski",
        metric_params: dict[str, Any] | None = None,
        kernel: str = "gaussian",
        **kwargs,
    ) -> None:
        self._detector_kwargs = {
            "contamination": float(contamination),
            "bandwidth": float(bandwidth),
            "algorithm": str(algorithm),
            "leaf_size": int(leaf_size),
            "metric": str(metric),
            "metric_params": dict(metric_params) if metric_params is not None else None,
            "kernel": str(kernel),
            **kwargs,
        }

        super().__init__(contamination=contamination, feature_extractor=feature_extractor)

    def _build_detector(self):
        return CoreKDE(**self._detector_kwargs)

    def fit(self, X: Iterable[str], y=None):
        return super().fit(X, y=y)

    def decision_function(self, X):
        return super().decision_function(X)

