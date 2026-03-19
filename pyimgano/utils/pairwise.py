"""Small pairwise helpers for detectors.

We avoid pulling in heavy metric libraries for a handful of simple operations.
"""

from __future__ import annotations

import numpy as np


def pairwise_distances_no_broadcast(x, y):
    """Row-wise Euclidean distance between matching rows of `x` and `y`.

    This matches the typical use in reconstruction-error style detectors where
    `x[i]` should be compared only to `y[i]` (not all-pairs).

    Parameters
    ----------
    x, y:
        Array-like with the same shape `(n_samples, n_features)`.

    Returns
    -------
    distances : ndarray of shape (n_samples,)
    """

    x = np.asarray(x)
    y = np.asarray(y)
    if x.shape != y.shape:
        raise ValueError(f"x and y must have the same shape, got {x.shape} vs {y.shape}")
    if x.ndim != 2:
        raise ValueError(f"Expected 2D arrays, got ndim={x.ndim}")

    diff = x - y
    return np.sqrt(np.sum(diff * diff, axis=1))
