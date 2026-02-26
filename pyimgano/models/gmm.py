# -*- coding: utf-8 -*-
"""
GMM (Gaussian Mixture Model) detector.

We fit a Gaussian mixture model and use negative log-likelihood (NLL) as the
outlier score: higher NLL => more anomalous.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.utils import check_array

from .baseml import BaseVisionDetector
from .registry import register_model


class CoreGMM:
    """Sklearn-backed GMM anomaly detector."""

    def __init__(
        self,
        *,
        contamination: float = 0.1,
        n_components: int = 1,
        covariance_type: str = "full",
        tol: float = 1e-3,
        reg_covar: float = 1e-6,
        max_iter: int = 100,
        n_init: int = 1,
        init_params: str = "kmeans",
        weights_init=None,
        means_init=None,
        precisions_init=None,
        random_state: Optional[int] = None,
        warm_start: bool = False,
        **kwargs: Any,
    ) -> None:
        self.contamination = float(contamination)
        self.gmm_kwargs = dict(
            n_components=int(n_components),
            covariance_type=str(covariance_type),
            tol=float(tol),
            reg_covar=float(reg_covar),
            max_iter=int(max_iter),
            n_init=int(n_init),
            init_params=str(init_params),
            weights_init=weights_init,
            means_init=means_init,
            precisions_init=precisions_init,
            random_state=random_state,
            warm_start=bool(warm_start),
            **dict(kwargs),
        )

        self.gmm_: GaussianMixture | None = None
        self.decision_scores_: np.ndarray | None = None

    def fit(self, X, y=None):  # noqa: ANN001, ANN201 - sklearn/pyod-like API
        X = check_array(X, ensure_2d=True, dtype=np.float64)
        self.gmm_ = GaussianMixture(**self.gmm_kwargs)
        self.gmm_.fit(X)
        self.decision_scores_ = self.decision_function(X)
        return self

    def decision_function(self, X):  # noqa: ANN001, ANN201 - sklearn/pyod-like API
        if self.gmm_ is None:
            raise RuntimeError("Detector must be fitted before calling decision_function")
        X = check_array(X, ensure_2d=True, dtype=np.float64)
        log_likelihood = self.gmm_.score_samples(X)
        return (-log_likelihood).astype(np.float64).ravel()


@register_model(
    "vision_gmm",
    tags=("vision", "classical", "gmm", "density", "baseline"),
    metadata={
        "description": "Gaussian Mixture Model detector (density baseline)",
        "type": "density",
    },
)
class VisionGMM(BaseVisionDetector):
    """Vision-compatible GMM detector."""

    def __init__(
        self,
        *,
        feature_extractor=None,
        contamination: float = 0.1,
        n_components: int = 1,
        covariance_type: str = "full",
        tol: float = 1e-3,
        reg_covar: float = 1e-6,
        max_iter: int = 100,
        n_init: int = 1,
        init_params: str = "kmeans",
        weights_init=None,
        means_init=None,
        precisions_init=None,
        random_state: Optional[int] = None,
        warm_start: bool = False,
        **kwargs: Any,
    ) -> None:
        self._detector_kwargs = {
            "contamination": float(contamination),
            "n_components": int(n_components),
            "covariance_type": str(covariance_type),
            "tol": float(tol),
            "reg_covar": float(reg_covar),
            "max_iter": int(max_iter),
            "n_init": int(n_init),
            "init_params": str(init_params),
            "weights_init": weights_init,
            "means_init": means_init,
            "precisions_init": precisions_init,
            "random_state": random_state,
            "warm_start": bool(warm_start),
            **dict(kwargs),
        }
        super().__init__(contamination=contamination, feature_extractor=feature_extractor)

    def _build_detector(self):
        return CoreGMM(**self._detector_kwargs)

    def fit(self, X: Iterable[str], y=None):
        return super().fit(X, y=y)

    def decision_function(self, X):
        return super().decision_function(X)

