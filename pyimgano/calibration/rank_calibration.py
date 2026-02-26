"""Unsupervised rank calibration for anomaly scores.

Many detectors output scores on different scales. A lightweight way to
standardize them without supervised labels is to map scores to their empirical
CDF position under the *training* score distribution.
"""

from __future__ import annotations

import numpy as np


class RankCalibrator:
    """Empirical-CDF score calibrator (train -> sorted scores)."""

    def __init__(self) -> None:
        self._sorted_train: np.ndarray | None = None

    def fit(self, train_scores) -> "RankCalibrator":
        x = np.asarray(train_scores, dtype=np.float64).reshape(-1)
        self._sorted_train = np.sort(x, kind="mergesort")
        return self

    def transform(self, scores) -> np.ndarray:
        if self._sorted_train is None:
            raise RuntimeError("RankCalibrator is not fitted yet. Call fit() first.")

        x = np.asarray(scores, dtype=np.float64).reshape(-1)
        n = int(self._sorted_train.shape[0])
        if n <= 0:
            return np.zeros_like(x, dtype=np.float64)

        # Empirical CDF: P(train <= x) with "right" tie handling.
        ranks = np.searchsorted(self._sorted_train, x, side="right").astype(np.float64)
        return ranks / float(n)

    def fit_transform(self, train_scores) -> np.ndarray:
        self.fit(train_scores)
        return self.transform(train_scores)
