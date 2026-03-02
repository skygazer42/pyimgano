# -*- coding: utf-8 -*-
"""Calibrated cosine kNN detector for embedding feature matrices.

Industrial motivation
---------------------
Deep embedding backbones often produce *useful* anomaly scores but on
inconsistent scales across:
- categories
- cameras / lighting
- backbones

This wrapper makes the score scale more stable by applying an unsupervised
standardization (default: empirical CDF / "rank" -> [0,1]) on top of the
`core_knn_cosine` distance baseline.

Score convention: **higher score => more anomalous**.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from pyimgano.calibration.score_standardization import ScoreStandardizer
from pyimgano.utils.fitted import require_fitted

from .base_detector import BaseDetector
from .core_knn_cosine import CoreKNNCosineModel
from .registry import register_model


@register_model(
    "core_knn_cosine_calibrated",
    tags=("classical", "core", "features", "neighbors", "knn", "cosine", "calibration"),
    metadata={
        "description": "Cosine kNN detector with unsupervised score standardization (rank/zscore/robust/minmax)",
        "input": "features",
        "wrapper": True,
    },
)
class CoreKNNCosineCalibrated(BaseDetector):
    """Cosine kNN detector with score standardization.

    Notes
    -----
    - This is *not* a supervised calibration; it is fitted only on training
      scores.
    - With `method='rank'`, scores are in [0,1] and thresholding becomes
      quantile-stable.
    """

    def __init__(
        self,
        *,
        contamination: float = 0.1,
        # Core kNN cosine knobs
        n_neighbors: int = 5,
        knn_method: str = "largest",
        normalize: bool = True,
        eps: float = 1e-12,
        n_jobs: int = 1,
        # Standardization knobs
        method: str = "rank",
        standardize_eps: float = 1e-12,
        random_state: Optional[int] = None,  # API compatibility (unused)
    ) -> None:
        super().__init__(contamination=float(contamination))
        self.n_neighbors = int(n_neighbors)
        self.knn_method = str(knn_method)
        self.normalize = bool(normalize)
        self.eps = float(eps)
        self.n_jobs = int(n_jobs)
        self.method = str(method)
        self.standardize_eps = float(standardize_eps)
        self.random_state = random_state

        self.base_model_ = None
        self.standardizer_ = None

    def fit(self, X, y=None):  # noqa: ANN001, ANN201 - sklearn-like API
        base = CoreKNNCosineModel(
            contamination=float(self.contamination),
            n_neighbors=int(self.n_neighbors),
            method=str(self.knn_method),
            normalize=bool(self.normalize),
            eps=float(self.eps),
            n_jobs=int(self.n_jobs),
            random_state=self.random_state,
        )
        base.fit(X, y=y)

        train_scores = np.asarray(getattr(base, "decision_scores_", base.decision_function(X)), dtype=np.float64).reshape(
            -1
        )
        std = ScoreStandardizer(method=str(self.method), eps=float(self.standardize_eps)).fit(
            train_scores
        )

        self.base_model_ = base
        self.standardizer_ = std

        self.decision_scores_ = std.transform(train_scores).astype(np.float64, copy=False)
        self._process_decision_scores()
        self._set_n_classes(y)
        return self

    def decision_function(self, X):  # noqa: ANN001, ANN201 - sklearn-like API
        require_fitted(self, ["base_model_", "standardizer_"])
        base: CoreKNNCosineModel = self.base_model_  # type: ignore[assignment]
        std: ScoreStandardizer = self.standardizer_  # type: ignore[assignment]
        raw = np.asarray(base.decision_function(X), dtype=np.float64).reshape(-1)
        return std.transform(raw)


__all__ = ["CoreKNNCosineCalibrated"]

