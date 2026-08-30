# -*- coding: utf-8 -*-
"""
LOCI (Local Correlation Integral) detector.

LOCI is effective for detecting individual outliers and micro-clusters by
analyzing neighborhood densities across multiple radii.

Reference:
    Papadimitriou, S., Kitagawa, H., Gibbons, P.B. and Faloutsos, C., 2003.
    LOCI: Fast Outlier Detection Using the Local Correlation Integral.
"""

# UPSTREAM: yzhao062/pyod @ 34f7996effac700a5166d882d5e94c6e6078fae3 (BSD-2-Clause; adapted)

from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.utils import check_array

from .baseml import BaseVisionDetector
from .core_feature_base import CoreFeatureDetector
from .registry import register_model


class CoreLOCI:
    """LOCI with a deterministic per-query novelty extension.

    Training scores retain the paper's dataset-level semantics.  New samples
    are scored one at a time against the fitted reference set so the result is
    independent of unrelated samples in the same API call.
    """

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
        if not 0.0 < self.alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {self.alpha}.")
        if self.k <= 0.0:
            raise ValueError(f"k must be positive, got {self.k}.")

        self.decision_scores_: np.ndarray | None = None
        self._X_fit: np.ndarray | None = None
        self._fit_sorted_distances: np.ndarray | None = None

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

    def _calculate_scores(self, x: np.ndarray) -> np.ndarray:
        outlier_scores = np.zeros(x.shape[0], dtype=np.float64)
        dist_matrix = squareform(pdist(x, metric="euclidean")).astype(np.float64)
        sorted_distances = np.sort(dist_matrix, axis=1)
        max_dist = float(np.max(dist_matrix)) if dist_matrix.size else 0.0
        r_max = max_dist / float(self.alpha)

        for p_ix in range(x.shape[0]):
            critical_values = self._critical_values(dist_matrix, p_ix, r_max)
            if critical_values.size == 0:
                continue
            thresholds = critical_values * self.alpha
            count_grid = np.vstack(
                [np.searchsorted(row, thresholds, side="left") for row in sorted_distances]
            )

            for column, r in enumerate(critical_values):
                sample = np.nonzero(dist_matrix[p_ix] <= r)[0]
                if sample.size == 0:
                    continue

                n_values = count_grid[sample, column]
                cur_alpha_n = float(count_grid[p_ix, column])
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

    def _score_query(self, row: np.ndarray) -> float:
        if self._X_fit is None or self._fit_sorted_distances is None:
            raise RuntimeError("Detector must be fitted before scoring queries")

        query_distances = np.linalg.norm(self._X_fit - row.reshape(1, -1), axis=1)
        positive = query_distances[query_distances > 0.0]
        if positive.size == 0:
            return 0.0
        critical_values = np.sort(np.concatenate((positive, positive / self.alpha))).astype(
            np.float64, copy=False
        )
        thresholds = critical_values * self.alpha
        count_grid = np.vstack(
            [np.searchsorted(row, thresholds, side="left") for row in self._fit_sorted_distances]
        )
        sorted_query_distances = np.sort(query_distances)

        score = 0.0
        for column, radius in enumerate(critical_values):
            sample = np.flatnonzero(query_distances <= radius)
            threshold = thresholds[column]

            training_counts = count_grid[sample, column].copy()
            # The augmented reference contains the query once. It contributes
            # to a training point's alpha-neighborhood exactly when that point
            # is closer than the current threshold.
            training_counts += query_distances[sample] < threshold
            query_count = int(np.searchsorted(sorted_query_distances, threshold, side="left")) + 1
            n_values = np.append(training_counts, query_count)

            n_hat = float(np.mean(n_values))
            if n_hat <= 0.0:
                continue
            mdef = 1.0 - (float(query_count) / n_hat)
            sigma_mdef = float(np.std(n_values) / n_hat)
            if n_hat >= 20 and sigma_mdef > 0.0:
                score = float(mdef / sigma_mdef)
                if mdef > (self.k * sigma_mdef):
                    break
        return score

    def fit(self, x, _y=None):  # noqa: ANN001, ANN201 - sklearn-like API
        del _y
        x = check_array(x, ensure_2d=True, dtype=np.float64)
        self._X_fit = np.asarray(x, dtype=np.float64).copy()
        fit_distances = squareform(pdist(self._X_fit, metric="euclidean")).astype(np.float64)
        self._fit_sorted_distances = np.sort(fit_distances, axis=1)
        self.decision_scores_ = self._calculate_scores(x)
        return self

    def decision_function(self, x):  # noqa: ANN001, ANN201 - sklearn-like API
        if (
            self.decision_scores_ is None
            or self._X_fit is None
            or self._fit_sorted_distances is None
        ):
            raise RuntimeError("Detector must be fitted before calling decision_function")
        x = check_array(x, ensure_2d=True, dtype=np.float64)
        if x.shape[1] != self._X_fit.shape[1]:
            raise ValueError(f"Expected {self._X_fit.shape[1]} features, got {x.shape[1]}")
        return np.asarray([self._score_query(row) for row in x], dtype=np.float64)


@register_model(
    "core_loci",
    tags=("classical", "core", "features", "loci", "density"),
    metadata={
        "description": "Core LOCI detector on feature matrices (native wrapper)",
        "input": "features",
        "paper": "Papadimitriou et al., 2003",
        "paper_fidelity": "paper-adaptation",
        "implementation_status": "paper-transductive-core-per-query-novelty-extension",
        "year": 2003,
    },
)
class CoreLOCIModel(CoreFeatureDetector):
    """Core (feature-matrix) LOCI detector with BaseDetector thresholding."""

    def __init__(
        self,
        *,
        contamination: float = 0.1,
        alpha: float = 0.5,
        k: float = 3.0,
        **kwargs,
    ) -> None:
        self._backend_kwargs = dict(
            contamination=float(contamination),
            alpha=float(alpha),
            k=float(k),
            **dict(kwargs),
        )
        super().__init__(contamination=float(contamination))

    def _build_detector(self):
        return CoreLOCI(**self._backend_kwargs)


@register_model(
    "vision_loci",
    tags=("vision", "classical", "loci"),
    metadata={
        "description": "Vision wrapper for LOCI with per-query novelty extension",
        "paper_fidelity": "paper-adaptation",
        "implementation_status": "paper-transductive-core-per-query-novelty-extension",
    },
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

    def fit(self, x: Iterable[str], y=None):
        return super().fit(x, y=y)

    def decision_function(self, x):
        return super().decision_function(x)
