"""Unsupervised score standardization helpers.

Many anomaly detectors produce scores on different scales. In industrial
pipelines (and in ensembles) it is often useful to standardize scores in a
label-free way.

Supported methods:
- `rank`: empirical CDF position under training scores -> [0,1]
- `zscore`: (x - mean) / std
- `robust_zscore`: (x - median) / MAD
- `minmax`: (x - min) / (max - min) -> [0,1]

All transformations preserve the convention:
  higher score => more anomalous
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .rank_calibration import RankCalibrator


@dataclass
class ScoreStandardizer:
    method: str = "rank"
    eps: float = 1e-12

    def __post_init__(self) -> None:
        self.method = str(self.method).strip().lower()
        self.eps = float(self.eps)
        self._rank: RankCalibrator | None = None
        self._mu: float | None = None
        self._sigma: float | None = None
        self._med: float | None = None
        self._mad: float | None = None
        self._min: float | None = None
        self._max: float | None = None

    def fit(self, train_scores) -> "ScoreStandardizer":
        x = np.asarray(train_scores, dtype=np.float64).reshape(-1)
        m = self.method

        if m == "rank":
            self._rank = RankCalibrator().fit(x)
            return self

        if m == "zscore":
            self._mu = float(np.mean(x)) if x.size else 0.0
            self._sigma = float(np.std(x)) if x.size else 1.0
            if not np.isfinite(self._sigma) or self._sigma <= 0.0:
                self._sigma = 1.0
            return self

        if m in {"robust_zscore", "robust", "mad"}:
            self._med = float(np.median(x)) if x.size else 0.0
            mad = float(np.median(np.abs(x - self._med))) if x.size else 1.0
            if not np.isfinite(mad) or mad <= 0.0:
                mad = 1.0
            self._mad = mad
            return self

        if m == "minmax":
            if x.size:
                self._min = float(np.min(x))
                self._max = float(np.max(x))
            else:
                self._min = 0.0
                self._max = 1.0
            return self

        raise ValueError(f"Unknown standardization method: {self.method!r}")

    def transform(self, scores) -> np.ndarray:
        x = np.asarray(scores, dtype=np.float64).reshape(-1)
        m = self.method

        if m == "rank":
            if self._rank is None:
                raise RuntimeError("ScoreStandardizer(method='rank') is not fitted.")
            return np.asarray(self._rank.transform(x), dtype=np.float64).reshape(-1)

        if m == "zscore":
            if self._mu is None or self._sigma is None:
                raise RuntimeError("ScoreStandardizer(method='zscore') is not fitted.")
            return ((x - float(self._mu)) / max(float(self._sigma), float(self.eps))).astype(
                np.float64
            )

        if m in {"robust_zscore", "robust", "mad"}:
            if self._med is None or self._mad is None:
                raise RuntimeError("ScoreStandardizer(method='robust_zscore') is not fitted.")
            return ((x - float(self._med)) / max(float(self._mad), float(self.eps))).astype(
                np.float64
            )

        if m == "minmax":
            if self._min is None or self._max is None:
                raise RuntimeError("ScoreStandardizer(method='minmax') is not fitted.")
            lo = float(self._min)
            hi = float(self._max)
            denom = hi - lo
            if not np.isfinite(denom) or denom <= 0.0:
                return np.zeros_like(x, dtype=np.float64)
            z = (x - lo) / denom
            return np.clip(z, 0.0, 1.0).astype(np.float64)

        raise ValueError(f"Unknown standardization method: {self.method!r}")

    def fit_transform(self, train_scores) -> np.ndarray:
        self.fit(train_scores)
        return self.transform(train_scores)
