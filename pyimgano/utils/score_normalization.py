"""Score normalization utilities.

Many ensembles and calibration heuristics want to combine heterogeneous anomaly
scores. This module provides lightweight, dependency-stable normalization
methods.
"""

from __future__ import annotations

import numpy as np


def _as_1d(scores) -> np.ndarray:
    arr = np.asarray(scores, dtype=np.float64)
    if arr.ndim != 1:
        return arr.reshape(-1)
    return arr


def zscore(scores, *, eps: float = 1e-12) -> np.ndarray:
    """Zero-mean, unit-variance normalization."""

    x = _as_1d(scores)
    mu = float(np.mean(x))
    sigma = float(np.std(x))
    return (x - mu) / (sigma + float(eps))


def minmax(scores, *, eps: float = 1e-12) -> np.ndarray:
    """Map scores to [0, 1] via min-max scaling."""

    x = _as_1d(scores)
    lo = float(np.min(x))
    hi = float(np.max(x))
    return (x - lo) / (hi - lo + float(eps))


def rank01(scores) -> np.ndarray:
    """Map scores to [0, 1] by empirical CDF (rank normalization).

    Ties are handled by assigning the average rank among equal values.
    """

    x = _as_1d(scores)
    n = int(x.shape[0])
    if n <= 1:
        return np.zeros_like(x, dtype=np.float64)

    order = np.argsort(x, kind="mergesort")  # stable for deterministic ties
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64)

    # Tie handling: average ranks for equal values.
    sorted_x = x[order]
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        if j - i > 1:
            avg = float(np.mean(ranks[order[i:j]]))
            ranks[order[i:j]] = avg
        i = j

    return ranks / float(n - 1)


def quantile(scores) -> np.ndarray:
    """Alias for empirical CDF normalization.

    This is intentionally simple (no sklearn QuantileTransformer) to keep the
    runtime surface minimal.
    """

    return rank01(scores)


def normalize(scores, method: str) -> np.ndarray:
    """Normalize scores with a named method."""

    m = str(method).lower().strip()
    if m in {"z", "zscore", "standard", "standardize"}:
        return zscore(scores)
    if m in {"minmax", "min-max", "mm"}:
        return minmax(scores)
    if m in {"rank", "rank01", "cdf"}:
        return rank01(scores)
    if m in {"quantile", "quant"}:
        return quantile(scores)
    raise ValueError(f"Unknown normalization method: {method!r}")

