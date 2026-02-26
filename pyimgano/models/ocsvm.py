# -*- coding: utf-8 -*-
"""One-Class SVM (OCSVM) detector.

This is a native PyImgAno implementation built on scikit-learn's
`sklearn.svm.OneClassSVM` (no `pyod` dependency).

Notes
-----
- scikit-learn's `decision_function` returns larger values for inliers.
  We negate it so that **higher score => more anomalous**, matching PyOD /
  PyImgAno conventions.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
from numpy.typing import NDArray
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.utils import check_array

from .baseml import BaseVisionDetector
from .registry import register_model


class CoreOCSVM:
    """OCSVM core on feature vectors."""

    def __init__(
        self,
        *,
        contamination: float = 0.1,
        kernel: str = "rbf",
        nu: Optional[float] = None,
        gamma: str | float = "scale",
        degree: int = 3,
        coef0: float = 0.0,
        tol: float = 1e-3,
        shrinking: bool = True,
        cache_size: float = 200.0,
        max_iter: int = -1,
        preprocessing: bool = True,
    ) -> None:
        self.contamination = float(contamination)
        self.kernel = str(kernel)
        self.nu = float(nu) if nu is not None else None
        self.gamma = gamma
        self.degree = int(degree)
        self.coef0 = float(coef0)
        self.tol = float(tol)
        self.shrinking = bool(shrinking)
        self.cache_size = float(cache_size)
        self.max_iter = int(max_iter)
        self.preprocessing = bool(preprocessing)

        self.scaler_: StandardScaler | None = None
        self.estimator_: OneClassSVM | None = None
        self.n_features_in_: int | None = None
        self.decision_scores_: NDArray[np.float64] | None = None

    def fit(self, X, y=None):  # noqa: ANN001, ANN201 - sklearn/pyod-like API
        X = check_array(X, ensure_2d=True, dtype=np.float64)
        if X.shape[0] == 0:
            raise ValueError("Training set cannot be empty")
        self.n_features_in_ = int(X.shape[1])

        if self.preprocessing:
            self.scaler_ = StandardScaler()
            X_train = self.scaler_.fit_transform(X)
        else:
            X_train = X

        # `nu` is a natural mapping of expected outlier fraction for OCSVM.
        nu = float(self.contamination) if self.nu is None else float(self.nu)
        if not (0.0 < nu <= 1.0):
            raise ValueError(f"nu must be in (0, 1], got {nu}")

        self.estimator_ = OneClassSVM(
            kernel=self.kernel,
            nu=nu,
            gamma=self.gamma,
            degree=self.degree,
            coef0=self.coef0,
            tol=self.tol,
            shrinking=self.shrinking,
            cache_size=self.cache_size,
            max_iter=self.max_iter,
        )
        self.estimator_.fit(X_train)

        self.decision_scores_ = np.asarray(self.decision_function(X), dtype=np.float64)
        return self

    def decision_function(self, X):  # noqa: ANN001, ANN201 - sklearn/pyod-like API
        if self.estimator_ is None or self.n_features_in_ is None:
            raise RuntimeError("Detector must be fitted before calling decision_function")

        X = check_array(X, ensure_2d=True, dtype=np.float64)
        if int(X.shape[1]) != int(self.n_features_in_):
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X.shape[1]}"
            )

        if self.preprocessing and self.scaler_ is not None:
            X_eval = self.scaler_.transform(X)
        else:
            X_eval = X

        # sklearn: positive => inlier. We flip the sign.
        scores = -np.asarray(self.estimator_.decision_function(X_eval), dtype=np.float64).reshape(-1)
        return scores


@register_model(
    "vision_ocsvm",
    tags=("vision", "classical", "svm", "one-class", "ocsvm"),
    metadata={
        "description": "One-Class SVM outlier detector (sklearn backend)",
    },
)
class VisionOCSVM(BaseVisionDetector):
    """Vision-compatible One-Class SVM detector."""

    def __init__(
        self,
        *,
        feature_extractor=None,
        contamination: float = 0.1,
        kernel: str = "rbf",
        nu: Optional[float] = None,
        gamma: str | float = "scale",
        degree: int = 3,
        coef0: float = 0.0,
        tol: float = 1e-3,
        shrinking: bool = True,
        cache_size: float = 200.0,
        max_iter: int = -1,
        preprocessing: bool = True,
    ) -> None:
        self._detector_kwargs = dict(
            contamination=float(contamination),
            kernel=str(kernel),
            nu=nu,
            gamma=gamma,
            degree=int(degree),
            coef0=float(coef0),
            tol=float(tol),
            shrinking=bool(shrinking),
            cache_size=float(cache_size),
            max_iter=int(max_iter),
            preprocessing=bool(preprocessing),
        )
        super().__init__(contamination=contamination, feature_extractor=feature_extractor)

    def _build_detector(self):
        return CoreOCSVM(**self._detector_kwargs)

    def fit(self, X: Iterable[str], y=None):
        return super().fit(X, y=y)

    def decision_function(self, X):
        return super().decision_function(X)

