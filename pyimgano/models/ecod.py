# -*- coding: utf-8 -*-
"""
ECOD (Empirical Cumulative Distribution-based Outlier Detection).

ECOD is a parameter-free, highly interpretable outlier detection algorithm
based on empirical CDF functions.

Reference:
    Li, Z., Zhao, Y., Botta, N., Ionescu, C. and Hu, X., 2022.
    ECOD: Unsupervised Outlier Detection Using Empirical Cumulative
    Distribution Functions. IEEE Transactions on Knowledge and Data
    Engineering (TKDE).
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from numpy.typing import NDArray
from sklearn.utils import check_array

from .baseml import BaseVisionDetector
from .registry import register_model


def _skew_sign(X: NDArray[np.floating]) -> NDArray[np.float64]:
    """Compute per-feature skewness sign using a stable, dependency-free formula."""
    X = np.asarray(X, dtype=np.float64)
    mean = X.mean(axis=0)
    centered = X - mean
    m2 = np.mean(centered**2, axis=0)
    m3 = np.mean(centered**3, axis=0)
    denom = np.power(m2, 1.5)
    skew = np.divide(m3, denom, out=np.zeros_like(m3), where=denom > 0.0)
    return np.sign(skew).astype(np.float64)


class CoreECOD:
    """Pure NumPy + sklearn-style implementation of ECOD.

    Notes
    -----
    This implementation uses the empirical CDF computed on the training set and
    evaluates test points against that fixed distribution (no test leakage).
    """

    def __init__(self, *, contamination: float = 0.1, n_jobs: int = 1, eps: float = 1e-12):
        self.contamination = float(contamination)
        self.n_jobs = int(n_jobs)  # kept for API compatibility (currently unused)
        self.eps = float(eps)

        self._X_sorted: NDArray[np.float64] | None = None
        self._skew_sign: NDArray[np.float64] | None = None
        self.decision_scores_: NDArray[np.float64] | None = None

    def fit(self, X, y=None):  # noqa: ANN001, ANN201 - sklearn/pyod-like API
        X = check_array(X, ensure_2d=True, dtype=np.float64)
        self._X_sorted = np.sort(X, axis=0)
        self._skew_sign = _skew_sign(X)

        self.decision_scores_ = self.decision_function(X)
        return self

    def decision_function(self, X):  # noqa: ANN001, ANN201 - sklearn/pyod-like API
        if self._X_sorted is None or self._skew_sign is None:
            raise RuntimeError("Detector must be fitted before calling decision_function")

        X = check_array(X, ensure_2d=True, dtype=np.float64)
        n_train, n_features = self._X_sorted.shape
        if X.shape[1] != n_features:
            raise ValueError(
                f"Expected {n_features} features, got {X.shape[1]}"
            )

        # Compute feature-wise tail scores and sum.
        scores_by_feature = np.empty((X.shape[0], n_features), dtype=np.float64)

        for j in range(n_features):
            col_sorted = self._X_sorted[:, j]
            x_col = X[:, j]

            # Left-tail: P(X_train <= x)
            cdf_l = np.searchsorted(col_sorted, x_col, side="right") / float(n_train)
            # Right-tail: P(X_train >= x) = P(-X_train <= -x)
            cdf_r = (n_train - np.searchsorted(col_sorted, x_col, side="left")) / float(
                n_train
            )

            cdf_l = np.clip(cdf_l, self.eps, 1.0)
            cdf_r = np.clip(cdf_r, self.eps, 1.0)
            u_l = -np.log(cdf_l)
            u_r = -np.log(cdf_r)

            s = self._skew_sign[j]
            if s > 0:
                u_skew = u_r
            elif s < 0:
                u_skew = u_l
            else:
                u_skew = u_l + u_r

            o = np.maximum(u_l, u_r)
            o = np.maximum(u_skew, o)
            scores_by_feature[:, j] = o

        return scores_by_feature.sum(axis=1).ravel()


@register_model(
    "vision_ecod",
    tags=("vision", "classical", "ecod", "parameter-free", "high-performance"),
    metadata={
        "description": "ECOD - Empirical CDF-based outlier detector (TKDE 2022)",
        "paper": "Li et al., TKDE 2022",
        "year": 2022,
        "interpretable": True,
        "parameter_free": True,
        "fast": True,
        "benchmark_rank": "top-tier",
    },
)
class VisionECOD(BaseVisionDetector):
    """Vision-compatible ECOD detector for anomaly detection."""

    def __init__(
        self,
        *,
        feature_extractor=None,
        contamination: float = 0.1,
        n_jobs: int = 1,
        eps: float = 1e-12,
    ) -> None:
        self._detector_kwargs = {
            "contamination": float(contamination),
            "n_jobs": int(n_jobs),
            "eps": float(eps),
        }
        super().__init__(contamination=contamination, feature_extractor=feature_extractor)

    def _build_detector(self):
        return CoreECOD(**self._detector_kwargs)

    # Keep explicit type hints for path-like inputs (BaseVisionDetector handles the work).
    def fit(self, X: Iterable[str], y=None):
        try:
            is_empty = len(X) == 0  # type: ignore[arg-type]
        except TypeError:
            is_empty = False
        if is_empty:
            raise ValueError("Training set cannot be empty")

        return super().fit(X, y=y)

    def decision_function(self, X):
        return super().decision_function(X)
