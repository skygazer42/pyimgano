# -*- coding: utf-8 -*-
"""
QMCD (Wrap-around Quasi-Monte Carlo Discrepancy) detector.

QMCD measures how well each point "fills" a hypercube relative to other points
via a wrap-around discrepancy criterion. Points with larger discrepancy are
more likely to be outliers.

Reference:
    Fang, K.T., Hickernell, F.J. and Winker, P., 2001.
    Wrap-around L2-discrepancy of lattice rules.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import scipy.stats as stats
from numba import njit, prange
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import check_array

from .baseml import BaseVisionDetector
from .registry import register_model


@njit(fastmath=True, parallel=True)
def _wrap_around_discrepancy(data: np.ndarray, check: np.ndarray) -> np.ndarray:
    """Wrap-around Quasi-Monte Carlo discrepancy."""

    n = data.shape[0]
    d = data.shape[1]
    p = check.shape[0]

    disc = np.zeros(p, dtype=np.float64)
    for i in prange(p):
        dc = 0.0
        for j in prange(n):
            prod = 1.0
            for k in prange(d):
                x_kikj = abs(check[i, k] - data[j, k])
                prod *= 1.5 - x_kikj + x_kikj * x_kikj
            dc += prod
        disc[i] = dc

    return -((4.0 / 3.0) ** d) + (1.0 / (n**2)) * disc


class CoreQMCD:
    """Native QMCD core implementation."""

    def __init__(self, *, contamination: float = 0.1) -> None:
        self.contamination = float(contamination)

        self._scaler: MinMaxScaler | None = None
        self._fitted_data: np.ndarray | None = None
        self._is_flipped: bool = False

        self.decision_scores_: np.ndarray | None = None

    def fit(self, X, y=None):  # noqa: ANN001, ANN201 - sklearn/pyod-like API
        X = check_array(X, ensure_2d=True, dtype=np.float64)

        self._scaler = MinMaxScaler()
        X_norm = self._scaler.fit_transform(X)
        self._fitted_data = X_norm.copy()

        scores = _wrap_around_discrepancy(X_norm, X_norm)

        # Flip scores based on PyOD criterion so "higher = more anomalous"
        self._is_flipped = False
        skew = float(stats.skew(scores))
        kurt = float(stats.kurtosis(scores))
        if (skew < 0) or ((skew >= 0) and (kurt < 0)):
            scores = scores.max() + scores.min() - scores
            self._is_flipped = True

        self.decision_scores_ = np.asarray(scores, dtype=np.float64).ravel()
        return self

    def decision_function(self, X):  # noqa: ANN001, ANN201 - sklearn/pyod-like API
        if self.decision_scores_ is None or self._scaler is None or self._fitted_data is None:
            raise RuntimeError("Detector must be fitted before calling decision_function")

        X = check_array(X, ensure_2d=True, dtype=np.float64)
        X_norm = self._scaler.transform(X)
        scores = _wrap_around_discrepancy(self._fitted_data, X_norm)
        if self._is_flipped:
            scores = self.decision_scores_.max() + self.decision_scores_.min() - scores
        return np.asarray(scores, dtype=np.float64).ravel()


@register_model(
    "vision_qmcd",
    tags=("vision", "classical", "qmcd", "robust", "baseline"),
    metadata={
        "description": "QMCD wrap-around discrepancy detector (robust-statistical baseline)",
        "type": "robust-statistical",
    },
)
class VisionQMCD(BaseVisionDetector):
    """Vision-compatible QMCD detector."""

    def __init__(
        self,
        *,
        feature_extractor=None,
        contamination: float = 0.1,
    ) -> None:
        self._detector_kwargs = {"contamination": float(contamination)}
        super().__init__(contamination=contamination, feature_extractor=feature_extractor)

    def _build_detector(self):
        return CoreQMCD(**self._detector_kwargs)

    def fit(self, X: Iterable[str], y=None):
        return super().fit(X, y=y)

    def decision_function(self, X):
        return super().decision_function(X)

