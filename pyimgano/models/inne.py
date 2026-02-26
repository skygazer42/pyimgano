# -*- coding: utf-8 -*-
"""INNE (Isolation-based anomaly detection using nearest-neighbour ensembles).

INNE builds an ensemble of hypersphere sets by subsampling the training data.
Each hypersphere is centered at a subsampled point with radius equal to its
nearest-neighbour distance (within the subsample). A test point is considered
more anomalous if it is covered only by large hyperspheres (or none).

Reference
---------
Bandaragoda, T.R., Ting, K.M., Albrecht, D., Liu, F.T. and Wells, J.R., 2014.
Efficient anomaly detection by isolation using nearest neighbour ensemble.
IEEE International Conference on Data Mining Workshop.

Notes
-----
This implementation is based on the INNE algorithm and is compatible with the
PyOD-style detector contract used across PyImgAno, but is implemented natively
without importing `pyod`.
"""

from __future__ import annotations

import numbers
from warnings import warn

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import euclidean_distances
from sklearn.utils import check_array
from sklearn.utils.validation import check_random_state

from .baseml import BaseVisionDetector
from .registry import register_model

_MIN_FLOAT = np.finfo(float).eps
_MAX_INT = np.iinfo(np.int32).max


class CoreINNE:
    """Pure NumPy + sklearn implementation of INNE."""

    def __init__(
        self,
        *,
        contamination: float = 0.1,
        n_estimators: int = 200,
        max_samples: int | float | str = "auto",
        random_state=None,
    ) -> None:
        self.contamination = float(contamination)
        self.n_estimators = int(n_estimators)
        self.max_samples = max_samples
        self.random_state = random_state

        self.max_samples_: int | None = None
        self.decision_scores_: NDArray[np.float64] | None = None

    def fit(self, X, y=None):  # noqa: ANN001, ANN201 - sklearn/pyod-like API
        X = check_array(X, accept_sparse=False, dtype=np.float64)
        n_samples = int(X.shape[0])
        if n_samples < 2:
            raise ValueError("INNE requires at least 2 samples to fit")

        if self.n_estimators < 1:
            raise ValueError(f"n_estimators must be >= 1, got {self.n_estimators}")

        # Resolve max_samples following the PyOD convention.
        if isinstance(self.max_samples, str):
            if self.max_samples == "auto":
                max_samples = min(8, n_samples)
            else:
                raise ValueError(
                    f"max_samples {self.max_samples!r} is not supported. "
                    "Valid choices are: 'auto', int, or float."
                )
        elif isinstance(self.max_samples, numbers.Integral):
            if int(self.max_samples) > n_samples:
                warn(
                    f"max_samples ({self.max_samples}) is greater than n_samples ({n_samples}); "
                    "max_samples will be set to n_samples."
                )
                max_samples = n_samples
            else:
                max_samples = int(self.max_samples)
        else:  # float
            max_samples_f = float(self.max_samples)
            if not 0.0 < max_samples_f <= 1.0:
                raise ValueError(f"max_samples must be in (0, 1], got {self.max_samples!r}")
            max_samples = max(1, int(max_samples_f * n_samples))

        self.max_samples_ = int(max_samples)
        self._fit_ensemble(X)
        self.decision_scores_ = np.asarray(self.decision_function(X), dtype=np.float64)
        return self

    def _fit_ensemble(self, X: NDArray[np.float64]) -> None:
        n_samples, n_features = map(int, X.shape)
        assert self.max_samples_ is not None

        self._centroids = np.empty((self.n_estimators, self.max_samples_, n_features), dtype=np.float64)
        self._ratio = np.empty((self.n_estimators, self.max_samples_), dtype=np.float64)
        self._centroids_radius = np.empty((self.n_estimators, self.max_samples_), dtype=np.float64)

        rnd0 = check_random_state(self.random_state)
        self._seeds = rnd0.randint(_MAX_INT, size=self.n_estimators)

        for i in range(self.n_estimators):
            rnd = check_random_state(self._seeds[i])
            center_index = rnd.choice(n_samples, self.max_samples_, replace=False)
            centroids = X[center_index]
            self._centroids[i] = centroids

            # Pairwise centroid distances (squared).
            center_dist = euclidean_distances(centroids, centroids, squared=True)
            np.fill_diagonal(center_dist, np.inf)

            # Radius of each hypersphere is the nearest-neighbour distance of the centroid.
            centroid_radius = np.amin(center_dist, axis=1)
            self._centroids_radius[i] = centroid_radius

            cnn_index = np.argmin(center_dist, axis=1)
            cnn_radius = centroid_radius[cnn_index]

            self._ratio[i] = 1.0 - (cnn_radius + _MIN_FLOAT) / (centroid_radius + _MIN_FLOAT)

    def decision_function(self, X):  # noqa: ANN001, ANN201 - sklearn/pyod-like API
        if self.max_samples_ is None or not hasattr(self, "_centroids"):
            raise RuntimeError("Detector must be fitted before calling decision_function")

        X = check_array(X, accept_sparse=False, dtype=np.float64)
        assert self.max_samples_ is not None

        isolation_scores = np.ones((self.n_estimators, X.shape[0]), dtype=np.float64)

        # Each test instance is evaluated against n_estimators sets of hyperspheres.
        for i in range(self.n_estimators):
            x_dists = euclidean_distances(X, self._centroids[i], squared=True)

            cover_radius = np.where(
                x_dists <= self._centroids_radius[i][None, :],
                self._centroids_radius[i][None, :],
                np.nan,
            )
            covered_mask = ~np.isnan(cover_radius).all(axis=1)
            if not np.any(covered_mask):
                continue

            # Pick, for each covered x, the centroid of the smallest hypersphere covering it.
            cnn_x = np.nanargmin(cover_radius[covered_mask], axis=1)
            isolation_scores[i, covered_mask] = self._ratio[i][cnn_x]

        # The isolation scores are averaged to produce the anomaly score.
        scores = np.mean(isolation_scores, axis=0).reshape(-1)
        return scores.astype(np.float64, copy=False)


@register_model(
    "vision_inne",
    tags=("vision", "classical", "isolation", "inne", "fast"),
    metadata={
        "description": "INNE - Isolation using Nearest-Neighbor Ensembles (ICDMW 2014)",
        "paper": "Bandaragoda et al., ICDM 2014",
        "year": 2014,
        "fast": True,
        "scalable": True,
    },
)
class VisionINNE(BaseVisionDetector):
    """Vision-compatible INNE detector for anomaly detection."""

    def __init__(
        self,
        *,
        feature_extractor=None,
        contamination: float = 0.1,
        n_estimators: int = 200,
        max_samples: int | float | str = "auto",
        random_state=None,
    ) -> None:
        self._detector_kwargs = dict(
            contamination=float(contamination),
            n_estimators=int(n_estimators),
            max_samples=max_samples,
            random_state=random_state,
        )
        super().__init__(contamination=contamination, feature_extractor=feature_extractor)

    def _build_detector(self):
        return CoreINNE(**self._detector_kwargs)

    def fit(self, X: Iterable[str], y=None):
        return super().fit(X, y=y)

    def decision_function(self, X):
        return super().decision_function(X)
