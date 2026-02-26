# -*- coding: utf-8 -*-
"""
SOS (Stochastic Outlier Selection).

SOS quantifies "outlierness" as the probability that a point is *not* selected
as a neighbor by any other point (based on affinities that are tuned to a
target perplexity).

Reference:
    Janssens, J.H.M., Huszár, F., Postma, E. and van den Herik, H.J., 2012.
    Stochastic Outlier Selection. (AIDA)

Notes
-----
This implementation is inspired by the original SOS algorithm and aligns with
the common PyOD-style API used across this codebase (decision scores where
higher = more anomalous).
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.utils import check_array

from .baseml import BaseVisionDetector
from .registry import register_model


def _get_perplexity(D: np.ndarray, beta: float) -> tuple[float, np.ndarray]:
    """Compute entropy (H) and the unnormalized affinity row for a given beta."""

    A = np.exp(-D * beta)
    sumA = np.sum(A)
    # numerical stability: sumA can be extremely small in high beta regimes
    if not np.isfinite(sumA) or sumA <= 0.0:
        return float("nan"), A
    H = float(np.log(sumA) + beta * np.sum(D * A) / sumA)
    return H, A


class CoreSOS:
    """Pure NumPy implementation of SOS (Stochastic Outlier Selection)."""

    def __init__(
        self,
        *,
        contamination: float = 0.1,
        perplexity: float = 4.5,
        metric: str = "euclidean",
        eps: float = 1e-5,
    ) -> None:
        self.contamination = float(contamination)
        self.perplexity = float(perplexity)
        self.metric = str(metric)
        self.eps = float(eps)

        self.decision_scores_: np.ndarray | None = None

    def _x2d(self, X: np.ndarray) -> np.ndarray:
        (n, d) = X.shape
        metric = self.metric.lower()
        if metric == "none":
            if n != d:
                raise ValueError(
                    "If metric='none', X must be a square dissimilarity matrix."
                )
            return X

        if metric == "euclidean":
            sumX = np.sum(np.square(X), axis=1)
            D = np.sqrt(np.abs(np.add(np.add(-2 * np.dot(X, X.T), sumX).T, sumX)))
            return D

        # Fallback to scipy for other metrics (already a dependency of pyimgano).
        from scipy.spatial import distance  # type: ignore

        return distance.squareform(distance.pdist(X, metric))

    def _d2a(self, D: np.ndarray) -> np.ndarray:
        (n, _) = D.shape
        A = np.zeros((n, n), dtype=np.float64)
        beta = np.ones((n,), dtype=np.float64)
        logU = float(np.log(self.perplexity))

        for i in range(n):
            betamin = -np.inf
            betamax = np.inf
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            Di = D[i, mask]
            H, thisA = _get_perplexity(Di, float(beta[i]))

            Hdiff = H - logU
            tries = 0
            while (np.isnan(Hdiff) or abs(Hdiff) > self.eps) and tries < 5000:
                if np.isnan(Hdiff):
                    beta[i] = beta[i] / 10.0
                elif Hdiff > 0:
                    betamin = float(beta[i])
                    if np.isinf(betamax):
                        beta[i] = beta[i] * 2.0
                    else:
                        beta[i] = (beta[i] + betamax) / 2.0
                else:
                    betamax = float(beta[i])
                    if np.isinf(betamin):
                        beta[i] = beta[i] / 2.0
                    else:
                        beta[i] = (beta[i] + betamin) / 2.0

                H, thisA = _get_perplexity(Di, float(beta[i]))
                Hdiff = H - logU
                tries += 1

            A[i, mask] = thisA.astype(np.float64, copy=False)

        return A

    def _a2b(self, A: np.ndarray) -> np.ndarray:
        row_sums = A.sum(axis=1, keepdims=True)
        # row_sums should be > 0, but guard anyway.
        row_sums = np.where(row_sums > 0.0, row_sums, 1.0)
        return A / row_sums

    def _b2o(self, B: np.ndarray) -> np.ndarray:
        return np.prod(1.0 - B, axis=0)

    def fit(self, X, y=None):  # noqa: ANN001, ANN201 - sklearn/pyod-like API
        X = check_array(X, ensure_2d=True, dtype=np.float64)
        if X.shape[0] < 2:
            self.decision_scores_ = np.zeros(X.shape[0], dtype=np.float64)
            return self

        if not (1.0 <= self.perplexity <= float(X.shape[0] - 1)):
            raise ValueError(
                f"perplexity must be in [1, n_samples-1], got {self.perplexity}"
            )

        D = self._x2d(X)
        A = self._d2a(D)
        B = self._a2b(A)
        O = self._b2o(B)
        self.decision_scores_ = np.asarray(O, dtype=np.float64).ravel()
        return self

    def decision_function(self, X):  # noqa: ANN001, ANN201 - sklearn/pyod-like API
        if self.decision_scores_ is None:
            raise RuntimeError("Detector must be fitted before calling decision_function")

        X = check_array(X, ensure_2d=True, dtype=np.float64)
        if X.shape[0] < 2:
            return np.zeros(X.shape[0], dtype=np.float64)

        D = self._x2d(X)
        A = self._d2a(D)
        B = self._a2b(A)
        O = self._b2o(B)
        return np.asarray(O, dtype=np.float64).ravel()


@register_model(
    "vision_sos",
    tags=("vision", "classical", "sos", "probabilistic", "baseline"),
    metadata={
        "description": "Stochastic Outlier Selection (probabilistic baseline)",
        "type": "probabilistic",
    },
)
class VisionSOS(BaseVisionDetector):
    """Vision-compatible SOS detector."""

    def __init__(
        self,
        *,
        feature_extractor=None,
        contamination: float = 0.1,
        perplexity: float = 4.5,
        metric: str = "euclidean",
        eps: float = 1e-5,
    ) -> None:
        self._detector_kwargs = {
            "contamination": float(contamination),
            "perplexity": float(perplexity),
            "metric": str(metric),
            "eps": float(eps),
        }
        super().__init__(contamination=contamination, feature_extractor=feature_extractor)

    def _build_detector(self):
        return CoreSOS(**self._detector_kwargs)

    def fit(self, X: Iterable[str], y=None):
        return super().fit(X, y=y)

    def decision_function(self, X):
        return super().decision_function(X)

