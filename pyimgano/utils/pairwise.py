"""Small pairwise helpers for detectors.

We avoid pulling in heavy metric libraries for a handful of simple operations.
"""

from __future__ import annotations

import numpy as np


def pairwise_distances_no_broadcast(X, Y):
    """Row-wise Euclidean distance between matching rows of `X` and `Y`.

    This matches the typical use in reconstruction-error style detectors where
    `X[i]` should be compared only to `Y[i]` (not all-pairs).

    Parameters
    ----------
    X, Y:
        Array-like with the same shape `(n_samples, n_features)`.

    Returns
    -------
    distances : ndarray of shape (n_samples,)
    """

    X = np.asarray(X)
    Y = np.asarray(Y)
    if X.shape != Y.shape:
        raise ValueError(f"X and Y must have the same shape, got {X.shape} vs {Y.shape}")
    if X.ndim != 2:
        raise ValueError(f"Expected 2D arrays, got ndim={X.ndim}")

    diff = X - Y
    return np.sqrt(np.sum(diff * diff, axis=1))

