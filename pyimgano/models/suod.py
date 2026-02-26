# -*- coding: utf-8 -*-
"""SUOD (Scalable Unsupervised Outlier Detection) - simplified native ensemble.

The original SUOD is an acceleration framework. In PyImgAno, we provide a
dependency-minimal, stable ensemble wrapper that:

- fits a list of base detectors on feature vectors
- standardizes their scores
- combines them via average or maximization

Reference
---------
Zhao, Y. et al., 2021. SUOD: Accelerating Large-scale Unsupervised Outlier
Detection. (various venues / arXiv versions)

Notes
-----
This implementation does *not* depend on the external `suod` package and does
not implement random projections / approximation. It focuses on producing a
robust ensemble score under the PyImgAno detector contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.utils import check_array

from .baseml import BaseVisionDetector
from .registry import register_model


@dataclass
class _ZScoreScaler:
    mean_: NDArray[np.float64]
    scale_: NDArray[np.float64]

    @classmethod
    def fit(cls, X: NDArray[np.float64]) -> "_ZScoreScaler":
        mu = np.mean(X, axis=0)
        sigma = np.std(X, axis=0)
        sigma = np.where(sigma > 0.0, sigma, 1.0)
        return cls(mean_=mu.astype(np.float64, copy=False), scale_=sigma.astype(np.float64, copy=False))

    def transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        return (X - self.mean_) / self.scale_


def _combine(score_mat: NDArray[np.float64], combination: str) -> NDArray[np.float64]:
    if score_mat.size == 0:
        return np.zeros((0,), dtype=np.float64)

    if combination == "average":
        return np.mean(score_mat, axis=1)
    if combination in {"max", "maximization"}:
        return np.max(score_mat, axis=1)
    raise ValueError("combination must be one of {'average', 'max'}")


class CoreSUOD:
    """Core SUOD-like ensemble operating on feature matrices."""

    def __init__(
        self,
        *,
        base_estimators: Optional[Sequence[object]] = None,
        contamination: float = 0.1,
        combination: str = "average",
        random_state: Optional[int] = None,
        standardize: str = "zscore",
        n_jobs=None,
        **kwargs,
    ) -> None:
        # Keep forward-compat: accept extra kwargs from the PyOD/SUOD API
        # surface without failing, but do not pretend to implement them.
        self._unused_kwargs = dict(kwargs)

        self.base_estimators = list(base_estimators) if base_estimators is not None else None
        self.contamination = float(contamination)
        self.combination = str(combination)
        self.random_state = random_state
        self.standardize = str(standardize)
        self.n_jobs = n_jobs

        self._scaler: _ZScoreScaler | None = None
        self.decision_scores_: NDArray[np.float64] | None = None

    def _default_base_estimators(self) -> List[object]:
        # Import lazily to avoid circular imports at module import time.
        from .copod import CoreCOPOD
        from .hbos import CoreHBOS
        from .iforest import CoreIForest
        from .knn import CoreKNN

        rs = self.random_state

        # Mirror the common "diverse + cheap" SUOD default list.
        return [
            CoreKNN(n_neighbors=15, method="largest"),
            CoreKNN(n_neighbors=20, method="largest"),
            CoreHBOS(n_bins=10),
            CoreHBOS(n_bins=20),
            CoreCOPOD(),
            CoreIForest(n_estimators=50, random_state=rs),
            CoreIForest(n_estimators=100, random_state=rs),
            CoreIForest(n_estimators=150, random_state=rs),
        ]

    def fit(self, X, y=None):  # noqa: ANN001, ANN201 - sklearn/pyod-like API
        X = check_array(X, ensure_2d=True, dtype=np.float64)
        n_samples = int(X.shape[0])
        if n_samples == 0:
            raise ValueError("Training set cannot be empty")

        estimators = self.base_estimators
        if estimators is None:
            estimators = self._default_base_estimators()

        if len(estimators) < 2:
            raise ValueError("base_estimators must contain at least 2 detectors")

        # Fit each base estimator and collect training scores.
        score_mat = np.zeros((n_samples, len(estimators)), dtype=np.float64)
        for i, est in enumerate(estimators):
            fit = getattr(est, "fit", None)
            decision = getattr(est, "decision_function", None)
            if not callable(fit) or not callable(decision):
                raise TypeError("Each base estimator must implement fit() and decision_function()")
            est.fit(X)
            scores = getattr(est, "decision_scores_", None)
            if scores is None:
                raise RuntimeError("Base estimator did not set decision_scores_ during fit")
            s = np.asarray(scores, dtype=np.float64).reshape(-1)
            if s.shape[0] != n_samples:
                raise ValueError("Base estimator returned unexpected decision_scores_ length")
            score_mat[:, i] = s

        if self.standardize != "zscore":
            raise ValueError("standardize must be 'zscore' for now")

        self._scaler = _ZScoreScaler.fit(score_mat)
        score_mat_norm = self._scaler.transform(score_mat)

        self._estimators_ = list(estimators)
        self.decision_scores_ = _combine(score_mat_norm, self.combination).astype(np.float64, copy=False)
        return self

    def decision_function(self, X):  # noqa: ANN001, ANN201 - sklearn/pyod-like API
        if self.decision_scores_ is None or self._scaler is None or not hasattr(self, "_estimators_"):
            raise RuntimeError("Detector must be fitted before calling decision_function")

        X = check_array(X, ensure_2d=True, dtype=np.float64)
        n_samples = int(X.shape[0])

        estimators = self._estimators_
        score_mat = np.zeros((n_samples, len(estimators)), dtype=np.float64)
        for i, est in enumerate(estimators):
            score_mat[:, i] = np.asarray(est.decision_function(X), dtype=np.float64).reshape(-1)

        score_mat_norm = self._scaler.transform(score_mat)
        return _combine(score_mat_norm, self.combination).astype(np.float64, copy=False)


@register_model(
    "vision_suod",
    tags=("vision", "classical", "ensemble", "suod"),
    metadata={"description": "SUOD-style score ensemble (native, simplified)"},
)
class VisionSUOD(BaseVisionDetector):
    """Vision-friendly SUOD wrapper using project feature extractors."""

    def __init__(
        self,
        *,
        feature_extractor=None,
        contamination: float = 0.1,
        base_estimators: Optional[Sequence[object]] = None,
        combination: str = "average",
        random_state: Optional[int] = None,
        standardize: str = "zscore",
        n_jobs=None,
        **kwargs,
    ) -> None:
        self._detector_kwargs = dict(
            base_estimators=base_estimators,
            contamination=float(contamination),
            combination=str(combination),
            random_state=random_state,
            standardize=str(standardize),
            n_jobs=n_jobs,
            **dict(kwargs),
        )
        super().__init__(contamination=contamination, feature_extractor=feature_extractor)

    def _build_detector(self):
        return CoreSUOD(**self._detector_kwargs)

    def fit(self, X: Iterable[str], y=None):
        return super().fit(X, y=y)

    def decision_function(self, X):
        return super().decision_function(X)

