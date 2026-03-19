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
the common sklearn-style API used across this codebase (decision scores where
higher = more anomalous).
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.utils import check_array

from .baseml import BaseVisionDetector
from .core_feature_base import CoreFeatureDetector
from .registry import register_model


def _get_perplexity(d: np.ndarray, beta: float) -> tuple[float, np.ndarray]:
    """Compute entropy (H) and the unnormalized affinity row for a given beta."""

    a = np.exp(-d * beta)
    sum_a = np.sum(a)
    # numerical stability: sumA can be extremely small in high beta regimes
    if not np.isfinite(sum_a) or sum_a <= 0.0:
        return float("nan"), a
    h = float(np.log(sum_a) + beta * np.sum(d * a) / sum_a)
    return h, a


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

    def _x2d(self, x: np.ndarray) -> np.ndarray:
        n, d = x.shape
        metric = self.metric.lower()
        if metric == "none":
            if n != d:
                raise ValueError("If metric='none', X must be a square dissimilarity matrix.")
            return x

        if metric == "euclidean":
            sum_x = np.sum(np.square(x), axis=1)
            d = np.sqrt(np.abs(np.add(np.add(-2 * np.dot(x, x.T), sum_x).T, sum_x)))
            return d

        # Fallback to scipy for other metrics (already a dependency of pyimgano).
        from scipy.spatial import distance  # type: ignore

        return distance.squareform(distance.pdist(x, metric))

    def _d2a(self, d: np.ndarray) -> np.ndarray:
        n, _ = d.shape
        a = np.zeros((n, n), dtype=np.float64)
        beta = np.ones((n,), dtype=np.float64)
        log_u = float(np.log(self.perplexity))

        for i in range(n):
            betamin = -np.inf
            betamax = np.inf
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            di = d[i, mask]
            h, this_a = _get_perplexity(di, float(beta[i]))

            hdiff = h - log_u
            tries = 0
            while (np.isnan(hdiff) or abs(hdiff) > self.eps) and tries < 5000:
                if np.isnan(hdiff):
                    beta[i] = beta[i] / 10.0
                elif hdiff > 0:
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

                h, this_a = _get_perplexity(di, float(beta[i]))
                hdiff = h - log_u
                tries += 1

            a[i, mask] = this_a.astype(np.float64, copy=False)

        return a

    def _a2b(self, a: np.ndarray) -> np.ndarray:
        row_sums = a.sum(axis=1, keepdims=True)
        # row_sums should be > 0, but guard anyway.
        row_sums = np.where(row_sums > 0.0, row_sums, 1.0)
        return a / row_sums

    def _b2o(self, b: np.ndarray) -> np.ndarray:
        return np.prod(1.0 - b, axis=0)

    def fit(self, x, y=None):  # noqa: ANN001, ANN201 - sklearn-like API
        del y
        x = check_array(x, ensure_2d=True, dtype=np.float64)
        if x.shape[0] < 2:
            self.decision_scores_ = np.zeros(x.shape[0], dtype=np.float64)
            return self

        if not (1.0 <= self.perplexity <= float(x.shape[0] - 1)):
            raise ValueError(f"perplexity must be in [1, n_samples-1], got {self.perplexity}")

        d = self._x2d(x)
        a = self._d2a(d)
        b = self._a2b(a)
        outlier_prob = self._b2o(b)
        self.decision_scores_ = np.asarray(outlier_prob, dtype=np.float64).ravel()
        return self

    def decision_function(self, x):  # noqa: ANN001, ANN201 - sklearn-like API
        if self.decision_scores_ is None:
            raise RuntimeError("Detector must be fitted before calling decision_function")

        x = check_array(x, ensure_2d=True, dtype=np.float64)
        if x.shape[0] < 2:
            return np.zeros(x.shape[0], dtype=np.float64)

        d = self._x2d(x)
        a = self._d2a(d)
        b = self._a2b(a)
        outlier_prob = self._b2o(b)
        return np.asarray(outlier_prob, dtype=np.float64).ravel()


@register_model(
    "core_sos",
    tags=("classical", "core", "features", "sos", "probabilistic"),
    metadata={
        "description": "SOS (Stochastic Outlier Selection) for feature matrices (native wrapper)",
        "type": "probabilistic",
    },
)
class CoreSOSDetector(CoreFeatureDetector):
    """Feature-matrix SOS detector (`core_*`).

    This wraps :class:`CoreSOS` into the native :class:`CoreFeatureDetector` contract:
    - `fit()` computes `decision_scores_` and derives `threshold_` via contamination
    - `decision_function()` returns scores (higher => more anomalous)
    """

    def __init__(
        self,
        *,
        contamination: float = 0.1,
        perplexity: float = 4.5,
        metric: str = "euclidean",
        eps: float = 1e-5,
    ) -> None:
        self.perplexity = float(perplexity)
        self.metric = str(metric)
        self.eps = float(eps)
        super().__init__(contamination=contamination)

    def _build_detector(self):  # noqa: ANN201
        return CoreSOS(
            contamination=float(self.contamination),
            perplexity=float(self.perplexity),
            metric=str(self.metric),
            eps=float(self.eps),
        )


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

    def fit(self, x: Iterable[str], y=None):
        return super().fit(x, y=y)

    def decision_function(self, x):
        return super().decision_function(x)
