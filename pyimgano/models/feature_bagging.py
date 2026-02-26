# -*- coding: utf-8 -*-
"""Feature Bagging (random feature subspace ensemble).

Feature Bagging improves stability by fitting multiple base detectors on
randomly sampled feature subspaces and aggregating their scores.

Reference
---------
Lazarevic, A. and Kumar, V., 2005. Feature bagging for outlier detection.
ACM SIGKDD.

Notes
-----
This is a native PyImgAno implementation (no `pyod` dependency). It is inspired
by the PyOD contract but implemented around the `pyimgano` detector API:

- core detectors operate on feature vectors (2D arrays)
- vision wrappers extract image features and delegate to the core
"""

from __future__ import annotations

import numbers
from typing import Iterable, List, Optional, Sequence, Union

import numpy as np
from numpy.typing import NDArray
from sklearn.neighbors import LocalOutlierFactor
from sklearn.utils import check_array, check_random_state

from ..utils.param_check import check_parameter
from .baseml import BaseVisionDetector
from .registry import register_model

_MAX_INT = int(np.iinfo(np.int32).max)


def _generate_feature_indices(
    rng: np.random.RandomState,
    *,
    bootstrap_features: bool,
    n_features: int,
    min_features: int,
    max_features: int,
) -> NDArray[np.int64]:
    """Randomly draw a feature subset.

    Parameters follow the PyOD/Sklearn bagging convention:
    - number of sampled features is drawn uniformly from [min_features, max_features]
    """

    if n_features < 1:
        raise ValueError("n_features must be >= 1")
    if min_features < 1:
        raise ValueError("min_features must be >= 1")
    if max_features < min_features:
        raise ValueError("max_features must be >= min_features")
    if max_features > n_features:
        raise ValueError("max_features must be <= n_features")

    random_n_features = int(rng.randint(min_features, max_features + 1))

    if bootstrap_features:
        return rng.randint(0, n_features, size=random_n_features, dtype=np.int64)

    return rng.choice(n_features, size=random_n_features, replace=False).astype(
        np.int64, copy=False
    )


class _CoreLOF:
    """Small LOF core used as default base estimator.

    We keep this private to avoid exposing a full LOF model surface here.
    """

    def __init__(self, *, n_neighbors: int = 20, n_jobs: int = 1) -> None:
        self.n_neighbors = int(n_neighbors)
        self.n_jobs = int(n_jobs)
        self.detector_: LocalOutlierFactor | None = None
        self.decision_scores_: NDArray[np.float64] | None = None

    def fit(self, X, y=None):  # noqa: ANN001, ANN201 - sklearn/pyod-like API
        X = check_array(X, ensure_2d=True, dtype=np.float64)
        n_samples = int(X.shape[0])
        if n_samples < 2:
            # Not enough points to estimate local density.
            self.detector_ = None
            self.decision_scores_ = np.zeros((n_samples,), dtype=np.float64)
            return self

        k = max(1, min(int(self.n_neighbors), n_samples - 1))
        self.detector_ = LocalOutlierFactor(
            n_neighbors=k,
            novelty=True,  # needed for scoring new samples
            contamination="auto",  # we threshold outside via BaseDetector
            n_jobs=self.n_jobs,
        )
        self.detector_.fit(X)

        # sklearn: negative_outlier_factor_ is lower (more negative) for outliers.
        self.decision_scores_ = (-np.asarray(self.detector_.negative_outlier_factor_, dtype=np.float64)).reshape(-1)
        return self

    def decision_function(self, X):  # noqa: ANN001, ANN201 - sklearn/pyod-like API
        if self.decision_scores_ is None:
            raise RuntimeError("Base estimator must be fitted before calling decision_function")

        X = check_array(X, ensure_2d=True, dtype=np.float64)
        if self.detector_ is None:
            return np.zeros((X.shape[0],), dtype=np.float64)

        # sklearn: score_samples higher => inlier. Negate to match "higher => more anomalous".
        return (-np.asarray(self.detector_.score_samples(X), dtype=np.float64)).reshape(-1)


class CoreFeatureBagging:
    """Feature bagging ensemble core operating on feature matrices."""

    def __init__(
        self,
        *,
        contamination: float = 0.1,
        n_estimators: int = 10,
        max_features: Union[int, float] = 1.0,
        bootstrap_features: bool = False,
        random_state: Optional[int] = None,
        combination: str = "average",
        base_estimator: str = "lof",
        n_jobs: int = 1,
        n_neighbors: int = 20,
    ) -> None:
        self.contamination = float(contamination)
        self.n_estimators = int(n_estimators)
        self.max_features = max_features
        self.bootstrap_features = bool(bootstrap_features)
        self.random_state = random_state
        self.combination = str(combination)
        self.base_estimator = str(base_estimator)
        self.n_jobs = int(n_jobs)
        self.n_neighbors = int(n_neighbors)

        self.n_features_in_: int | None = None
        self.estimators_: List[object] = []
        self.estimators_features_: List[NDArray[np.int64]] = []
        self.decision_scores_: NDArray[np.float64] | None = None

    def _make_base_estimator(self):
        if self.base_estimator == "lof":
            return _CoreLOF(n_neighbors=self.n_neighbors, n_jobs=self.n_jobs)
        raise ValueError(f"Unknown base_estimator: {self.base_estimator!r}. Supported: 'lof'")

    def fit(self, X, y=None):  # noqa: ANN001, ANN201 - sklearn/pyod-like API
        X = check_array(X, ensure_2d=True, dtype=np.float64)
        n_samples, n_features = map(int, X.shape)
        self.n_features_in_ = n_features

        if n_samples == 0:
            raise ValueError("Training set cannot be empty")

        if not isinstance(self.n_estimators, (numbers.Integral, np.integer)) or self.n_estimators < 1:
            raise ValueError("n_estimators must be an integer >= 1")

        # Feature bagging is meaningless with a single feature.
        check_parameter(n_features, low=2, include_left=True, param_name="n_features")

        min_features = max(1, int(0.5 * n_features))

        if isinstance(self.max_features, (numbers.Integral, np.integer)):
            max_features = int(self.max_features)
        else:
            max_features_f = float(self.max_features)
            if not (0.0 < max_features_f <= 1.0):
                raise ValueError("max_features as float must be in (0, 1]")
            max_features = int(max_features_f * n_features)

        check_parameter(
            max_features,
            low=min_features,
            high=n_features,
            include_left=True,
            include_right=True,
            param_name="max_features",
        )

        rng = check_random_state(self.random_state)
        seeds = rng.randint(_MAX_INT, size=self.n_estimators)

        self.estimators_ = []
        self.estimators_features_ = []
        for i in range(self.n_estimators):
            rnd = np.random.RandomState(int(seeds[i]))
            features = _generate_feature_indices(
                rnd,
                bootstrap_features=self.bootstrap_features,
                n_features=n_features,
                min_features=min_features,
                max_features=max_features,
            )
            estimator = self._make_base_estimator()
            estimator.fit(X[:, features])
            self.estimators_.append(estimator)
            self.estimators_features_.append(features)

        score_mat = self._get_train_score_matrix(n_samples=n_samples)
        self.decision_scores_ = self._combine_scores(score_mat).astype(np.float64, copy=False)
        return self

    def _get_train_score_matrix(self, *, n_samples: int) -> NDArray[np.float64]:
        mat = np.zeros((n_samples, len(self.estimators_)), dtype=np.float64)
        for i, est in enumerate(self.estimators_):
            scores = getattr(est, "decision_scores_", None)
            if scores is None:
                raise RuntimeError("Base estimator did not set decision_scores_ during fit")
            scores_np = np.asarray(scores, dtype=np.float64).reshape(-1)
            if scores_np.shape[0] != n_samples:
                raise ValueError("Base estimator returned unexpected decision_scores_ length")
            mat[:, i] = scores_np
        return mat

    def _combine_scores(self, score_mat: NDArray[np.float64]) -> NDArray[np.float64]:
        if score_mat.size == 0:
            return np.zeros((0,), dtype=np.float64)

        if self.combination == "average":
            return np.mean(score_mat, axis=1)
        if self.combination in {"max", "maximization"}:
            return np.max(score_mat, axis=1)

        raise ValueError("combination must be one of {'average', 'max'}")

    def decision_function(self, X):  # noqa: ANN001, ANN201 - sklearn/pyod-like API
        if self.n_features_in_ is None or self.decision_scores_ is None:
            raise RuntimeError("Detector must be fitted before calling decision_function")

        X = check_array(X, ensure_2d=True, dtype=np.float64)
        if int(X.shape[1]) != int(self.n_features_in_):
            raise ValueError(
                f"Number of features must match training data. "
                f"Model n_features={self.n_features_in_}, input n_features={X.shape[1]}."
            )

        score_mat = np.zeros((X.shape[0], len(self.estimators_)), dtype=np.float64)
        for i, (est, feats) in enumerate(zip(self.estimators_, self.estimators_features_)):
            score_mat[:, i] = np.asarray(est.decision_function(X[:, feats]), dtype=np.float64).reshape(-1)
        return self._combine_scores(score_mat).astype(np.float64, copy=False)


@register_model(
    "vision_feature_bagging",
    tags=("vision", "ensemble", "feature_bagging"),
    metadata={
        "description": "Feature Bagging - random feature-subspace ensemble (native)",
        "paper": "Lazarevic & Kumar, KDD 2005",
        "year": 2005,
        "ensemble": True,
        "robust": True,
    },
)
class VisionFeatureBagging(BaseVisionDetector):
    """Vision-compatible Feature Bagging detector for anomaly detection."""

    def __init__(
        self,
        *,
        feature_extractor=None,
        contamination: float = 0.1,
        n_estimators: int = 10,
        max_features: Union[int, float] = 1.0,
        bootstrap_features: bool = False,
        n_jobs: int = 1,
        random_state: Optional[int] = None,
        combination: str = "average",
        base_estimator: str = "lof",
        n_neighbors: int = 20,
    ) -> None:
        self._detector_kwargs = dict(
            contamination=float(contamination),
            n_estimators=int(n_estimators),
            max_features=max_features,
            bootstrap_features=bool(bootstrap_features),
            n_jobs=int(n_jobs),
            random_state=random_state,
            combination=str(combination),
            base_estimator=str(base_estimator),
            n_neighbors=int(n_neighbors),
        )
        super().__init__(contamination=contamination, feature_extractor=feature_extractor)

    def _build_detector(self):
        return CoreFeatureBagging(**self._detector_kwargs)

    def fit(self, X: Iterable[str], y=None):
        return super().fit(X, y=y)

    def decision_function(self, X):
        return super().decision_function(X)

