# -*- coding: utf-8 -*-
"""LSCP (Locally Selective Combination in Parallel) ensemble.

LSCP selects competent detectors in the *local region* of each test sample.

Reference
---------
Zhao, Y. et al., 2019. LSCP: Locally Selective Combination in Parallel Outlier
Ensembles. SDM (and related preprints).

Notes
-----
This is a native PyImgAno implementation inspired by the original LSCP design,
but implemented with minimal dependencies (NumPy + scikit-learn) and adapted
to the `pyimgano` detector contract.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable, List, Optional, Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.neighbors import KDTree
from sklearn.utils import check_array, check_random_state

from .baseml import BaseVisionDetector
from .registry import register_model

_MAX_INT = int(np.iinfo(np.int32).max)


def _pearson_corr(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.shape[0] != b.shape[0] or a.shape[0] < 2:
        return 0.0

    a0 = a - float(np.mean(a))
    b0 = b - float(np.mean(b))
    denom = float(np.sqrt(np.sum(a0 * a0) * np.sum(b0 * b0)))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(a0, b0) / denom)


def _zscore_standardize(
    train_scores: NDArray[np.float64],
    test_scores: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    mu = np.mean(train_scores, axis=0)
    sigma = np.std(train_scores, axis=0)
    sigma = np.where(sigma > 0.0, sigma, 1.0)
    return (train_scores - mu) / sigma, (test_scores - mu) / sigma


def _generate_feature_subspace(
    rng: np.random.RandomState,
    *,
    n_features: int,
    min_features: int,
    max_features: int,
) -> NDArray[np.int64]:
    if n_features < 1:
        raise ValueError("n_features must be >= 1")
    if min_features < 1:
        raise ValueError("min_features must be >= 1")
    if max_features < min_features:
        raise ValueError("max_features must be >= min_features")
    if max_features > n_features:
        raise ValueError("max_features must be <= n_features")

    k = int(rng.randint(min_features, max_features + 1))
    return rng.choice(n_features, size=k, replace=False).astype(np.int64, copy=False)


class CoreLSCP:
    """LSCP core operating on feature matrices."""

    def __init__(
        self,
        detector_list: Sequence[object],
        *,
        local_region_size: int = 30,
        local_max_features: float = 1.0,
        n_bins: int = 10,
        random_state: Optional[int] = None,
        contamination: float = 0.1,
        local_region_iterations: int = 20,
        local_min_features: float = 0.5,
    ) -> None:
        self.detector_list = list(detector_list)
        self.n_clf = int(len(self.detector_list))
        self.local_region_size = int(local_region_size)
        self.local_max_features = float(local_max_features)
        self.n_bins = int(n_bins)
        self.random_state = random_state
        self.contamination = float(contamination)
        self.local_region_iterations = int(local_region_iterations)
        self.local_min_features = float(local_min_features)

        self.X_train_: NDArray[np.float64] | None = None
        self.train_scores_: NDArray[np.float64] | None = None
        self.decision_scores_: NDArray[np.float64] | None = None

    def fit(self, X, y=None):  # noqa: ANN001, ANN201 - sklearn/pyod-like API
        if self.n_clf < 2:
            raise ValueError("detector_list must contain at least 2 base detectors")

        X = check_array(X, ensure_2d=True, dtype=np.float64)
        self.X_train_ = X

        train_scores = np.zeros((X.shape[0], self.n_clf), dtype=np.float64)
        for k, det in enumerate(self.detector_list):
            fit = getattr(det, "fit", None)
            decision = getattr(det, "decision_function", None)
            if not callable(fit) or not callable(decision):
                raise TypeError("Each base detector must implement fit() and decision_function()")

            det.fit(X)
            scores = getattr(det, "decision_scores_", None)
            if scores is None:
                raise RuntimeError("Base detector did not set decision_scores_ during fit")
            scores_np = np.asarray(scores, dtype=np.float64).reshape(-1)
            if scores_np.shape[0] != X.shape[0]:
                raise ValueError("Base detector returned unexpected decision_scores_ length")
            train_scores[:, k] = scores_np

        self.train_scores_ = train_scores
        self.decision_scores_ = np.asarray(self.decision_function(X), dtype=np.float64)
        return self

    def decision_function(self, X):  # noqa: ANN001, ANN201 - sklearn/pyod-like API
        if self.X_train_ is None or self.train_scores_ is None:
            raise RuntimeError("Detector must be fitted before calling decision_function")

        X = check_array(X, ensure_2d=True, dtype=np.float64)
        if X.shape[1] != self.X_train_.shape[1]:
            raise ValueError(
                f"Number of features must match training data. "
                f"Model n_features={self.X_train_.shape[1]}, input n_features={X.shape[1]}."
            )

        return self._get_decision_scores(X)

    def _get_decision_scores(self, X_test: NDArray[np.float64]) -> NDArray[np.float64]:
        assert self.X_train_ is not None
        assert self.train_scores_ is not None

        n_train, n_features = map(int, self.X_train_.shape)

        # Compute local regions for each test sample (indices into training set).
        local_regions = self._get_local_regions(X_test)

        # Collect test scores from all detectors.
        test_scores = np.zeros((X_test.shape[0], self.n_clf), dtype=np.float64)
        for k, det in enumerate(self.detector_list):
            test_scores[:, k] = np.asarray(det.decision_function(X_test), dtype=np.float64).reshape(-1)

        train_scores_norm, test_scores_norm = _zscore_standardize(self.train_scores_, test_scores)

        # Global pseudo-labels for training: max of standardized detector scores.
        training_pseudo = np.max(train_scores_norm, axis=1).reshape(-1)

        pred = np.zeros((X_test.shape[0],), dtype=np.float64)
        for i, region in enumerate(local_regions):
            region_idx = np.asarray(region, dtype=np.int64)
            if region_idx.size < 2:
                # Fallback: use global region.
                region_idx = np.arange(n_train, dtype=np.int64)

            local_pseudo = training_pseudo[region_idx]
            local_train_scores = train_scores_norm[region_idx, :]

            corr = np.zeros((self.n_clf,), dtype=np.float64)
            for d in range(self.n_clf):
                corr[d] = _pearson_corr(local_pseudo, local_train_scores[:, d])
            if np.isnan(corr).any():
                corr = np.nan_to_num(corr)

            competent = self._get_competent_detectors(corr)
            pred[i] = float(np.mean(test_scores_norm[i, competent]))

        return pred

    def _get_local_regions(self, X_test: NDArray[np.float64]) -> List[List[int]]:
        assert self.X_train_ is not None

        rng = check_random_state(self.random_state)

        n_train, n_features = map(int, self.X_train_.shape)
        if n_train == 0:
            raise ValueError("Empty training set")

        # Clamp parameters.
        local_region_size = int(max(1, min(self.local_region_size, n_train)))
        local_min = max(1, int(n_features * self.local_min_features))
        local_max = max(local_min, int(np.ceil(n_features * self.local_max_features)))
        local_max = min(local_max, n_features)

        # Collect neighbor indices across randomized subspaces.
        local_region_list: List[List[int]] = [[] for _ in range(X_test.shape[0])]

        for _ in range(self.local_region_iterations):
            feats = _generate_feature_subspace(
                rng, n_features=n_features, min_features=local_min, max_features=local_max
            )

            tree = KDTree(self.X_train_[:, feats], leaf_size=40)
            _, inds = tree.query(X_test[:, feats], k=local_region_size, return_distance=True)
            for j in range(X_test.shape[0]):
                local_region_list[j].extend([int(x) for x in inds[j]])

        threshold = int(self.local_region_iterations / 2)
        final_regions: List[List[int]] = []
        for j in range(X_test.shape[0]):
            counts = Counter(local_region_list[j])
            current = [idx for idx, c in counts.items() if c > threshold]

            # Ensure region is not degenerate.
            dec = 0
            while len(current) < 2 and threshold - dec > 0:
                dec += 1
                current = [idx for idx, c in counts.items() if c > (threshold - dec)]

            if not current:
                current = list(range(local_region_size))
            final_regions.append(current)

        return final_regions

    def _get_competent_detectors(self, corr_scores: NDArray[np.float64]) -> NDArray[np.int64]:
        scores = np.asarray(corr_scores, dtype=np.float64).reshape(-1)
        n_clf = int(scores.shape[0])

        n_bins = int(self.n_bins)
        if n_bins < 1:
            n_bins = 1
        if n_bins > n_clf:
            n_bins = n_clf

        hist, edges = np.histogram(scores, bins=n_bins)
        if hist.size == 0:
            return np.asarray([int(np.argmax(scores))], dtype=np.int64)

        max_bin = int(np.argmax(hist))
        lo = float(edges[max_bin])
        hi = float(edges[max_bin + 1])
        candidates = np.where((scores >= lo) & (scores <= hi))[0]
        if candidates.size == 0:
            candidates = np.asarray([int(np.argmax(scores))], dtype=np.int64)
        return candidates.astype(np.int64, copy=False)


@register_model(
    "vision_lscp",
    tags=("vision", "classical", "ensemble", "lscp"),
    metadata={"description": "LSCP - locally selective combination ensemble (native)"},
)
class VisionLSCP(BaseVisionDetector):
    """Vision-friendly LSCP wrapper using project feature extractors."""

    def __init__(
        self,
        *,
        feature_extractor=None,
        contamination: float = 0.1,
        detector_list: Sequence[object] | None = None,
        local_region_size: int = 30,
        local_max_features: float = 1.0,
        n_bins: int = 10,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> None:
        if detector_list is None:
            raise ValueError("VisionLSCP requires a non-empty 'detector_list' of base detectors.")

        if kwargs:
            # Keep forward-compat: accept extra kwargs without failing, but
            # avoid silently changing behavior by pretending to support them.
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"Unknown LSCP parameters: {unknown}")

        self._detector_kwargs = dict(
            detector_list=list(detector_list),
            local_region_size=int(local_region_size),
            local_max_features=float(local_max_features),
            n_bins=int(n_bins),
            random_state=random_state,
            contamination=float(contamination),
        )
        super().__init__(contamination=contamination, feature_extractor=feature_extractor)

    def _build_detector(self):
        return CoreLSCP(**self._detector_kwargs)

    def fit(self, X: Iterable[str], y=None):
        return super().fit(X, y=y)

    def decision_function(self, X):
        return super().decision_function(X)

