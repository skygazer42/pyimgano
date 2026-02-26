# -*- coding: utf-8 -*-
"""
LOCI (Local Correlation Integral) detector.

LOCI is effective for detecting individual outliers and micro-clusters by
analyzing neighborhood densities across multiple radii.

Reference:
    Papadimitriou, S., Kitagawa, H., Gibbons, P.B. and Faloutsos, C., 2003.
    LOCI: Fast Outlier Detection Using the Local Correlation Integral.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.utils import check_array

from .baseml import BaseVisionDetector
from .registry import register_model


class CoreLOCI:
    """Native LOCI implementation (Euclidean metric)."""

    def __init__(
        self,
        *,
        contamination: float = 0.1,
        alpha: float = 0.5,
        k: float = 3.0,
    ) -> None:
        self.contamination = float(contamination)
        self.alpha = float(alpha)
        self.k = float(k)

        self.decision_scores_: np.ndarray | None = None

    def _alpha_n(self, dist_matrix: np.ndarray, indices, r: float) -> np.ndarray:
        """Count alpha-neighborhood points (< r*alpha)."""

        threshold = float(r) * float(self.alpha)
        if isinstance(indices, (int, np.integer)):
            return np.count_nonzero(dist_matrix[int(indices)] < threshold)
        return np.count_nonzero(dist_matrix[indices] < threshold, axis=1)

    def _critical_values(self, dist_matrix: np.ndarray, p_ix: int, r_max: float) -> np.ndarray:
        distances = dist_matrix[p_ix]
        mask = (distances > 0.0) & (distances <= r_max)
        vals = distances[mask]
        if vals.size == 0:
            return np.asarray([], dtype=np.float64)
        return np.sort(np.concatenate([vals, vals / float(self.alpha)])).astype(np.float64)

    def _calculate_scores(self, X: np.ndarray) -> np.ndarray:
        outlier_scores = np.zeros(X.shape[0], dtype=np.float64)
        dist_matrix = squareform(pdist(X, metric="euclidean")).astype(np.float64)
        max_dist = float(np.max(dist_matrix)) if dist_matrix.size else 0.0
        r_max = max_dist / float(self.alpha) if self.alpha != 0 else max_dist

        for p_ix in range(X.shape[0]):
            critical_values = self._critical_values(dist_matrix, p_ix, r_max)
            if critical_values.size == 0:
                continue

            for r in critical_values:
                sample = np.nonzero(dist_matrix[p_ix] <= r)[0]
                if sample.size == 0:
                    continue

                n_values = self._alpha_n(dist_matrix, sample, float(r))
                cur_alpha_n = float(self._alpha_n(dist_matrix, p_ix, float(r)))
                n_hat = float(np.mean(n_values)) if n_values.size else 0.0
                if n_hat <= 0.0:
                    continue

                mdef = 1.0 - (cur_alpha_n / n_hat)
                sigma_mdef = float(np.std(n_values) / n_hat)

                if n_hat >= 20 and sigma_mdef > 0.0:
                    outlier_scores[p_ix] = float(mdef / sigma_mdef)
                    # Early break if point is clearly outlying at this radius.
                    if mdef > (self.k * sigma_mdef):
                        break

        return outlier_scores

    def fit(self, X, y=None):  # noqa: ANN001, ANN201 - sklearn/pyod-like API
        X = check_array(X, ensure_2d=True, dtype=np.float64)
        self.decision_scores_ = self._calculate_scores(X)
        return self

    def decision_function(self, X):  # noqa: ANN001, ANN201 - sklearn/pyod-like API
        if self.decision_scores_ is None:
            raise RuntimeError("Detector must be fitted before calling decision_function")
        X = check_array(X, ensure_2d=True, dtype=np.float64)
        return self._calculate_scores(X)


@register_model(
    "vision_loci",
    tags=("vision", "classical", "loci"),
    metadata={"description": "Vision wrapper for LOCI outlier detector (native)"},
)
class VisionLOCI(BaseVisionDetector):
    def __init__(
        self,
        *,
        feature_extractor=None,
        contamination: float = 0.1,
        alpha: float = 0.5,
        k: float = 3.0,
        **kwargs,
    ):
        self.detector_kwargs = dict(
            contamination=float(contamination),
            alpha=float(alpha),
            k=float(k),
            **dict(kwargs),
        )
        super().__init__(contamination=contamination, feature_extractor=feature_extractor)

    def _build_detector(self):
        return CoreLOCI(**self.detector_kwargs)

    def fit(self, X: Iterable[str], y=None):
        return super().fit(X, y=y)

    def decision_function(self, X):
        return super().decision_function(X)

