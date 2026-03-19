# -*- coding: utf-8 -*-
"""MCD (Minimum Covariance Determinant).

MCD is a robust estimator of covariance that can identify outliers based on the
Mahalanobis distance from a robust estimate.

Reference
---------
Rousseeuw, P.J. and Driessen, K.V., 1999. A fast algorithm for the minimum
covariance determinant estimator. Technometrics.

Notes
-----
This is a native implementation for PyImgAno using scikit-learn's
`sklearn.covariance.MinCovDet` (no extra outlier-toolbox dependency).
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
from numpy.typing import NDArray
from sklearn.covariance import MinCovDet
from sklearn.utils import check_array

from .baseml import BaseVisionDetector
from .core_feature_base import CoreFeatureDetector
from .registry import register_model


class CoreMCD:
    """Pure sklearn backend for robust covariance distances."""

    def __init__(
        self,
        *,
        contamination: float = 0.1,
        support_fraction: Optional[float] = None,
        random_state: Optional[int] = None,
        assume_centered: bool = False,
        max_features: int = 4096,
    ) -> None:
        self.contamination = float(contamination)
        self.support_fraction = support_fraction
        self.random_state = random_state
        self.assume_centered = bool(assume_centered)
        self.max_features = int(max_features)

        self.estimator_: MinCovDet | None = None
        self.n_features_in_: int | None = None
        self.decision_scores_: NDArray[np.float64] | None = None

    def fit(self, x, y=None):  # noqa: ANN001, ANN201 - sklearn-like API
        del y
        x = check_array(x, ensure_2d=True, dtype=np.float64)
        if x.shape[0] == 0:
            raise ValueError("Training set cannot be empty")

        _, n_features = map(int, x.shape)
        self.n_features_in_ = n_features

        # Robust covariance estimation is not meant for extremely high
        # dimensional features. Avoid allocating massive covariance matrices.
        if n_features > self.max_features:
            raise ValueError(
                "MCD is not suitable for extremely high-dimensional features. "
                f"Got n_features={n_features}, max_features={self.max_features}. "
                "Use a lower-dimensional feature extractor (recommended) or "
                "reduce dimensions before MCD (e.g., PCA)."
            )

        if self.support_fraction is not None and not (0.0 < float(self.support_fraction) <= 1.0):
            raise ValueError(f"support_fraction must be in (0, 1], got {self.support_fraction}")

        self.estimator_ = MinCovDet(
            support_fraction=self.support_fraction,
            random_state=self.random_state,
            assume_centered=self.assume_centered,
        )
        self.estimator_.fit(x)

        self.decision_scores_ = np.asarray(
            self.estimator_.mahalanobis(x), dtype=np.float64
        ).reshape(-1)
        return self

    def decision_function(self, x):  # noqa: ANN001, ANN201 - sklearn-like API
        if self.estimator_ is None or self.n_features_in_ is None:
            raise RuntimeError("Detector must be fitted before calling decision_function")

        x = check_array(x, ensure_2d=True, dtype=np.float64)
        if int(x.shape[1]) != int(self.n_features_in_):
            raise ValueError(f"Expected {self.n_features_in_} features, got {x.shape[1]}")

        return np.asarray(self.estimator_.mahalanobis(x), dtype=np.float64).reshape(-1)


@register_model(
    "core_mcd",
    tags=("classical", "core", "features", "statistical", "mcd", "robust"),
    metadata={
        "description": "Core MCD robust covariance outlier detector on feature matrices (native wrapper)",
        "input": "features",
        "paper": "Rousseeuw & Driessen, Technometrics 1999",
        "year": 1999,
        "robust": True,
    },
)
class CoreMCDModel(CoreFeatureDetector):
    """Core (feature-matrix) MCD detector with BaseDetector thresholding."""

    def __init__(
        self,
        *,
        contamination: float = 0.1,
        support_fraction: Optional[float] = None,
        random_state: Optional[int] = None,
        assume_centered: bool = False,
        max_features: int = 4096,
    ) -> None:
        self._backend_kwargs = {
            "contamination": float(contamination),
            "support_fraction": support_fraction,
            "random_state": random_state,
            "assume_centered": bool(assume_centered),
            "max_features": int(max_features),
        }
        super().__init__(contamination=float(contamination))

    def _build_detector(self):
        return CoreMCD(**self._backend_kwargs)


@register_model(
    "vision_mcd",
    tags=("vision", "classical", "statistical", "mcd", "robust"),
    metadata={
        "description": "MCD - Robust covariance-based outlier detector (MinCovDet backend)",
        "paper": "Rousseeuw & Driessen, Technometrics 1999",
        "year": 1999,
        "robust": True,
        "parametric": True,
    },
)
class VisionMCD(BaseVisionDetector):
    """Vision-compatible MCD detector for anomaly detection."""

    def __init__(
        self,
        *,
        feature_extractor=None,
        contamination: float = 0.1,
        support_fraction: Optional[float] = None,
        random_state: Optional[int] = None,
        assume_centered: bool = False,
        max_features: int = 4096,
    ) -> None:
        self._detector_kwargs = {
            "contamination": float(contamination),
            "support_fraction": support_fraction,
            "random_state": random_state,
            "assume_centered": bool(assume_centered),
            "max_features": int(max_features),
        }
        super().__init__(contamination=contamination, feature_extractor=feature_extractor)

    def _build_detector(self):
        return CoreMCD(**self._detector_kwargs)

    def fit(self, x: Iterable[str], y=None):
        return super().fit(x, y=y)

    def decision_function(self, x):
        return super().decision_function(x)
