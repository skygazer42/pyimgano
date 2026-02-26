# -*- coding: utf-8 -*-
"""Histogram-based Outlier Score (HBOS).

HBOS assumes feature independence and builds a 1D histogram per feature. The
outlier score of a sample is the sum of negative log-probabilities of its
feature values under those histograms.

Reference
---------
Goldstein, M. and Dengel, A., 2012. Histogram-based Outlier Score (HBOS): A fast
unsupervised anomaly detection algorithm. (in various workshop proceedings)

Notes
-----
This is a lightweight, dependency-minimal implementation inspired by PyOD's
HBOS contract, but implemented natively in PyImgAno (no `pyod` dependency).
"""

from __future__ import annotations

from typing import Iterable, List

import numpy as np
from numpy.typing import NDArray
from sklearn.utils import check_array

from .baseml import BaseVisionDetector
from .registry import register_model


class CoreHBOS:
    """Pure NumPy + sklearn-style implementation of HBOS."""

    def __init__(
        self,
        *,
        contamination: float = 0.1,
        n_bins: int = 10,
        alpha: float = 0.1,
        eps: float = 1e-12,
    ) -> None:
        self.contamination = float(contamination)
        self.n_bins = int(n_bins)
        self.alpha = float(alpha)
        self.eps = float(eps)

        self._bin_edges: List[NDArray[np.float64]] | None = None
        self._bin_log_prob: List[NDArray[np.float64]] | None = None
        self.n_features_in_: int | None = None
        self.decision_scores_: NDArray[np.float64] | None = None

    def _fit_feature(self, values: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if values.size == 0:
            raise ValueError("HBOS requires non-empty training data")

        # Constant features are common with sparse / quantized embeddings.
        # Build a single-bin histogram to avoid degenerate edges.
        if float(np.max(values) - np.min(values)) <= 0.0:
            center = float(values[0])
            edges = np.asarray([center - 0.5, center + 0.5], dtype=np.float64)
            counts = np.asarray([float(values.size)], dtype=np.float64)
        else:
            counts, edges = np.histogram(values, bins=self.n_bins)
            counts = counts.astype(np.float64, copy=False)

        # Additive smoothing to avoid log(0). This keeps the detector stable for
        # small sample sizes.
        counts = counts + self.alpha
        probs = counts / float(np.sum(counts))
        log_prob = -np.log(np.clip(probs, self.eps, 1.0))
        return edges.astype(np.float64, copy=False), log_prob.astype(np.float64, copy=False)

    def fit(self, X, y=None):  # noqa: ANN001, ANN201 - sklearn/pyod-like API
        X = check_array(X, ensure_2d=True, dtype=np.float64)
        if X.shape[0] == 0:
            raise ValueError("Training set cannot be empty")

        self.n_features_in_ = int(X.shape[1])
        self._bin_edges = []
        self._bin_log_prob = []
        for j in range(self.n_features_in_):
            edges, log_prob = self._fit_feature(X[:, j])
            self._bin_edges.append(edges)
            self._bin_log_prob.append(log_prob)

        self.decision_scores_ = np.asarray(self.decision_function(X), dtype=np.float64)
        return self

    def decision_function(self, X):  # noqa: ANN001, ANN201 - sklearn/pyod-like API
        if self._bin_edges is None or self._bin_log_prob is None or self.n_features_in_ is None:
            raise RuntimeError("Detector must be fitted before calling decision_function")

        X = check_array(X, ensure_2d=True, dtype=np.float64)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X.shape[1]}"
            )

        scores = np.zeros((X.shape[0],), dtype=np.float64)
        for j in range(self.n_features_in_):
            edges = self._bin_edges[j]
            log_prob = self._bin_log_prob[j]
            # Map each value to a histogram bin index.
            idx = np.searchsorted(edges, X[:, j], side="right") - 1
            idx = np.clip(idx, 0, log_prob.shape[0] - 1)
            scores += log_prob[idx]
        return scores.ravel()


@register_model(
    "vision_hbos",
    tags=("vision", "classical", "hbos", "histogram", "fast"),
    metadata={
        "description": "HBOS - Histogram-based Outlier Score (fast, interpretable baseline)",
        "interpretable": True,
        "fast": True,
    },
)
class VisionHBOS(BaseVisionDetector):
    """Vision-compatible HBOS detector for anomaly detection."""

    def __init__(
        self,
        *,
        feature_extractor=None,
        contamination: float = 0.1,
        n_bins: int = 10,
        alpha: float = 0.1,
        eps: float = 1e-12,
    ) -> None:
        self._detector_kwargs = dict(
            contamination=float(contamination),
            n_bins=int(n_bins),
            alpha=float(alpha),
            eps=float(eps),
        )
        super().__init__(contamination=contamination, feature_extractor=feature_extractor)

    def _build_detector(self):
        return CoreHBOS(**self._detector_kwargs)

    def fit(self, X: Iterable[str], y=None):
        return super().fit(X, y=y)

    def decision_function(self, X):
        return super().decision_function(X)
