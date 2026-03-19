# -*- coding: utf-8 -*-
"""LoOP (Local Outlier Probability).

This is a lightweight, native implementation that follows the same detector
contract as other classical models in PyImgAno:
- `fit(X)` stores `decision_scores_` for training data
- `decision_function(X)` returns 1D anomaly scores (higher = more anomalous)
- thresholding/labels are handled by `BaseDetector`
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.special import erf
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_array

from ..utils.fitted import require_fitted
from .base_detector import BaseDetector
from .baseml import BaseVisionDetector
from .registry import register_model

logger = logging.getLogger(__name__)


@register_model(
    "core_loop",
    tags=("classical", "core", "features", "neighbors", "probability"),
    metadata={
        "description": "LoOP - Local Outlier Probability (native implementation)",
        "paper": "Kriegel et al., CIKM 2009",
        "year": 2009,
    },
)
class CoreLoOP(BaseDetector):
    """Tabular LoOP detector."""

    def __init__(
        self,
        *,
        contamination: float = 0.1,
        n_neighbors: int = 10,
        lambda_: float = 3.0,
        metric: str = "minkowski",
        p: int = 2,
        n_jobs: int | None = None,
        eps: float = 1e-12,
    ) -> None:
        super().__init__(contamination=float(contamination))
        self.n_neighbors = int(n_neighbors)
        self.lambda_ = float(lambda_)
        self.metric = str(metric)
        self.p = int(p)
        self.n_jobs = n_jobs
        self.eps = float(eps)

    def fit(self, x, y=None):  # noqa: ANN001, ANN201 - sklearn-like API
        x_arr = check_array(x, ensure_2d=True, dtype=np.float64)
        self._set_n_classes(y)

        n = int(x_arr.shape[0])
        k = int(self.n_neighbors)
        if k <= 0:
            raise ValueError(f"n_neighbors must be > 0, got {self.n_neighbors}")
        if n <= k:
            raise ValueError(f"Need n_samples > n_neighbors, got n={n} k={k}")

        nn = NearestNeighbors(
            n_neighbors=k + 1,
            metric=self.metric,
            p=self.p,
            n_jobs=self.n_jobs,
        )
        nn.fit(x_arr)
        distances, indices = nn.kneighbors(x_arr, n_neighbors=k + 1, return_distance=True)

        # Discard self neighbor at position 0.
        d = np.asarray(distances[:, 1:], dtype=np.float64)
        nbr_idx = np.asarray(indices[:, 1:], dtype=np.int64)

        pdist = float(self.lambda_) * np.sqrt(np.mean(d * d, axis=1))
        mean_nbr_pdist = np.mean(pdist[nbr_idx], axis=1)
        plof = pdist / (mean_nbr_pdist + float(self.eps)) - 1.0

        nplof = float(self.lambda_) * float(np.std(plof)) + float(self.eps)
        loop = erf(plof / (nplof * float(np.sqrt(2.0))))
        loop = np.clip(loop, 0.0, 1.0)

        self._nn = nn
        self._X_train = x_arr
        self._pdist_train = pdist
        self._nplof = float(nplof)

        self.decision_scores_ = np.asarray(loop, dtype=np.float64).reshape(-1)
        self._process_decision_scores()
        return self

    def decision_function(self, x):  # noqa: ANN001, ANN201 - sklearn-like API
        require_fitted(self, ["_nn", "_pdist_train", "_nplof"])
        nn: NearestNeighbors = self._nn  # type: ignore[assignment]
        pdist_train = np.asarray(self._pdist_train, dtype=np.float64).reshape(-1)  # type: ignore[arg-type]
        nplof = float(self._nplof)  # type: ignore[arg-type]

        x_arr = check_array(x, ensure_2d=True, dtype=np.float64)
        k = int(self.n_neighbors)
        if k <= 0:
            raise ValueError(f"n_neighbors must be > 0, got {self.n_neighbors}")

        distances, indices = nn.kneighbors(x_arr, n_neighbors=k, return_distance=True)
        d = np.asarray(distances, dtype=np.float64)
        nbr_idx = np.asarray(indices, dtype=np.int64)

        pdist = float(self.lambda_) * np.sqrt(np.mean(d * d, axis=1))
        mean_nbr_pdist = np.mean(pdist_train[nbr_idx], axis=1)
        plof = pdist / (mean_nbr_pdist + float(self.eps)) - 1.0

        loop = erf(plof / (nplof * float(np.sqrt(2.0))))
        loop = np.clip(loop, 0.0, 1.0)
        return np.asarray(loop, dtype=np.float64).reshape(-1)


@register_model(
    "vision_loop",
    tags=("vision", "classical", "neighbors", "probability"),
    metadata={
        "description": "Vision LoOP - Local Outlier Probability",
        "paper": "Kriegel et al., CIKM 2009",
        "year": 2009,
    },
)
class VisionLoOP(BaseVisionDetector):
    """Vision wrapper for LoOP (runs on extracted features)."""

    def __init__(
        self,
        *,
        feature_extractor=None,
        contamination: float = 0.1,
        n_neighbors: int = 10,
        lambda_: float = 3.0,
        metric: str = "minkowski",
        p: int = 2,
        n_jobs: int | None = None,
        eps: float = 1e-12,
    ) -> None:
        self._detector_kwargs = {
            "contamination": float(contamination),
            "n_neighbors": int(n_neighbors),
            "lambda_": float(lambda_),
            "metric": str(metric),
            "p": int(p),
            "n_jobs": n_jobs,
            "eps": float(eps),
        }
        logger.debug("Initializing VisionLoOP with kwargs=%s", self._detector_kwargs)
        super().__init__(contamination=contamination, feature_extractor=feature_extractor)

    def _build_detector(self):
        return CoreLoOP(**self._detector_kwargs)
