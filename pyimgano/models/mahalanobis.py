# -*- coding: utf-8 -*-
"""Mahalanobis distance detector (mean + covariance).

This is a simple, surprisingly strong baseline for embeddings/features when
the normal data is approximately Gaussian in the feature space.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.utils.validation import check_array

from ..utils.fitted import require_fitted
from ._legacy_x import MISSING, resolve_legacy_x_keyword
from .base_detector import BaseDetector
from .baseml import BaseVisionDetector
from .registry import register_model

logger = logging.getLogger(__name__)


@register_model(
    "core_mahalanobis",
    tags=("classical", "core", "features", "distance", "gaussian"),
    metadata={"description": "Mahalanobis distance baseline (mean + covariance)"},
)
class CoreMahalanobis(BaseDetector):
    def __init__(
        self,
        *,
        contamination: float = 0.1,
        reg: float = 1e-6,
        max_features: int = 4096,
    ) -> None:
        super().__init__(contamination=float(contamination))
        self.reg = float(reg)
        self.max_features = int(max_features)

    def fit(self, x: object = MISSING, y=None, **kwargs: object):  # noqa: ANN001, ANN201
        x_arr = check_array(
            resolve_legacy_x_keyword(x, kwargs, method_name="fit"),
            ensure_2d=True,
            dtype=np.float64,
        )
        self._set_n_classes(y)

        if x_arr.shape[0] == 0:
            raise ValueError("X must be non-empty")

        n_features = int(x_arr.shape[1])
        if n_features > self.max_features:
            raise ValueError(
                "Mahalanobis is not suitable for extremely high-dimensional features. "
                f"Got n_features={n_features}, max_features={self.max_features}. "
                "Use a lower-dimensional feature extractor (recommended) or "
                "reduce dimensions before Mahalanobis (e.g., PCA / embeddings)."
            )

        mu = np.mean(x_arr, axis=0)
        diff = x_arr - mu
        cov = (diff.T @ diff) / max(1, x_arr.shape[0] - 1)
        cov = np.asarray(cov, dtype=np.float64)
        cov = cov + float(self.reg) * np.eye(cov.shape[0], dtype=np.float64)

        inv = np.linalg.pinv(cov)

        self.mean_ = mu
        self.cov_ = cov
        self.inv_cov_ = inv

        self.decision_scores_ = np.asarray(self.decision_function(x_arr), dtype=np.float64)
        self._process_decision_scores()
        return self

    def decision_function(self, x: object = MISSING, **kwargs: object):  # noqa: ANN001, ANN201
        require_fitted(self, ["mean_", "inv_cov_"])
        mu = np.asarray(self.mean_, dtype=np.float64)  # type: ignore[arg-type]
        inv = np.asarray(self.inv_cov_, dtype=np.float64)  # type: ignore[arg-type]

        x_arr = check_array(
            resolve_legacy_x_keyword(x, kwargs, method_name="decision_function"),
            ensure_2d=True,
            dtype=np.float64,
        )
        if x_arr.shape[1] != mu.shape[0]:
            raise ValueError(f"Expected {mu.shape[0]} features, got {x_arr.shape[1]}")

        diff = x_arr - mu
        md2 = np.einsum("ij,jk,ik->i", diff, inv, diff)
        return np.asarray(md2, dtype=np.float64).reshape(-1)


@register_model(
    "vision_mahalanobis",
    tags=("vision", "classical", "distance", "gaussian"),
    metadata={"description": "Vision Mahalanobis baseline (mean + covariance)"},
)
class VisionMahalanobis(BaseVisionDetector):
    def __init__(
        self,
        *,
        feature_extractor=None,
        contamination: float = 0.1,
        reg: float = 1e-6,
        max_features: int = 4096,
    ) -> None:
        self._detector_kwargs = {
            "contamination": float(contamination),
            "reg": float(reg),
            "max_features": int(max_features),
        }
        logger.debug("Initializing VisionMahalanobis with kwargs=%s", self._detector_kwargs)
        super().__init__(contamination=contamination, feature_extractor=feature_extractor)

    def _build_detector(self):
        return CoreMahalanobis(**self._detector_kwargs)
