# -*- coding: utf-8 -*-
"""RGraph (graph random-walk scoring) - simplified native implementation.

PyOD's RGraph implementation builds a transition matrix from a
self-representation model and derives anomaly scores from the stationary
distribution of a random walk.

For PyImgAno, we provide a lightweight graph-based detector that:

1. builds a kNN affinity graph on training features
2. runs a short random walk to estimate average visitation probabilities
3. uses the negative stationary probability as anomaly score

This keeps the *spirit* of RGraph (random walk on a robust graph) while
avoiding heavy dependencies and very expensive self-representation steps.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
from numpy.typing import NDArray
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_array

from .baseml import BaseVisionDetector
from .registry import register_model


class CoreRGraph:
    """Graph random-walk anomaly scoring core."""

    def __init__(
        self,
        *,
        contamination: float = 0.1,
        transition_steps: int = 10,
        n_nonzero: int = 10,
        gamma: float = 50.0,
        preprocessing: bool = True,
        metric: str = "minkowski",
        p: int = 2,
        eps: float = 1e-12,
        **kwargs,
    ) -> None:
        # Accept extra kwargs for forward-compat with the PyOD signature (e.g.
        # gamma_nz, tau, algorithm). They are unused in this simplified core.
        self._unused_kwargs = dict(kwargs)

        self.contamination = float(contamination)
        self.transition_steps = int(transition_steps)
        self.n_nonzero = int(n_nonzero)
        self.gamma = float(gamma)
        self.preprocessing = bool(preprocessing)
        self.metric = str(metric)
        self.p = int(p)
        self.eps = float(eps)

        self.scaler_: StandardScaler | None = None
        self._nn: NearestNeighbors | None = None
        self._train_X: NDArray[np.float64] | None = None
        self._neighbors: NDArray[np.int64] | None = None
        self._trans_weights: NDArray[np.float64] | None = None

        self.pi_: NDArray[np.float64] | None = None
        self.decision_scores_: NDArray[np.float64] | None = None

    def _affinity(self, distances: NDArray[np.float64]) -> NDArray[np.float64]:
        # Use an RBF-like kernel on squared distances.
        return np.exp(-self.gamma * np.square(distances, dtype=np.float64))

    def fit(self, X, y=None):  # noqa: ANN001, ANN201 - sklearn/pyod-like API
        X = check_array(X, ensure_2d=True, dtype=np.float64)
        n_samples = int(X.shape[0])
        if n_samples == 0:
            raise ValueError("Training set cannot be empty")

        if self.transition_steps < 1:
            raise ValueError("transition_steps must be >= 1")
        if self.n_nonzero < 1:
            raise ValueError("n_nonzero must be >= 1")
        if self.gamma <= 0.0:
            raise ValueError("gamma must be > 0")

        if self.preprocessing:
            self.scaler_ = StandardScaler()
            self._train_X = self.scaler_.fit_transform(X)
        else:
            self.scaler_ = None
            self._train_X = X

        k = min(self.n_nonzero + 1, n_samples)
        self._nn = NearestNeighbors(
            n_neighbors=k,
            algorithm="auto",
            metric=self.metric,
            p=self.p,
        )
        self._nn.fit(self._train_X)

        dists, inds = self._nn.kneighbors(self._train_X, n_neighbors=k, return_distance=True)
        if k > 1:
            dists = dists[:, 1:]
            inds = inds[:, 1:]
        else:
            dists = np.zeros((n_samples, 0), dtype=np.float64)
            inds = np.zeros((n_samples, 0), dtype=np.int64)

        weights = self._affinity(dists)

        # Normalize outgoing weights to create a row-stochastic transition.
        trans = np.zeros_like(weights, dtype=np.float64)
        for i in range(n_samples):
            row = weights[i]
            s = float(np.sum(row))
            if row.size == 0:
                continue
            if s <= self.eps:
                trans[i] = 1.0 / float(row.size)
            else:
                trans[i] = row / s

        self._neighbors = inds.astype(np.int64, copy=False)
        self._trans_weights = trans

        self.pi_ = self._random_walk_stationary(n_samples)
        self.decision_scores_ = (-self.pi_).astype(np.float64, copy=False)
        return self

    def _random_walk_stationary(self, n_samples: int) -> NDArray[np.float64]:
        if self._neighbors is None or self._trans_weights is None:
            raise RuntimeError("Internal error: missing transition data")

        pi = np.ones((n_samples,), dtype=np.float64) / float(n_samples)
        pi_bar = np.zeros((n_samples,), dtype=np.float64)

        neigh = self._neighbors
        w = self._trans_weights
        for _ in range(self.transition_steps):
            pi_next = np.zeros_like(pi)
            for i in range(n_samples):
                if neigh.shape[1] == 0:
                    continue
                pi_next[neigh[i]] += pi[i] * w[i]
            pi = pi_next
            pi_bar += pi

        pi_bar /= float(self.transition_steps)
        return pi_bar

    def decision_function(self, X):  # noqa: ANN001, ANN201 - sklearn/pyod-like API
        if self.pi_ is None or self._nn is None or self._train_X is None:
            raise RuntimeError("Detector must be fitted before calling decision_function")

        X = check_array(X, ensure_2d=True, dtype=np.float64)
        if self.preprocessing and self.scaler_ is not None:
            Xn = self.scaler_.transform(X)
        else:
            Xn = X

        n_train = int(self._train_X.shape[0])
        k = min(self.n_nonzero, n_train)
        if k <= 0:
            return np.zeros((Xn.shape[0],), dtype=np.float64)

        dists, inds = self._nn.kneighbors(Xn, n_neighbors=k, return_distance=True)
        weights = self._affinity(dists)

        # Approximate the stationary probability of a new point by its affinity
        # to training nodes weighted by their stationary probabilities.
        scores = np.zeros((Xn.shape[0],), dtype=np.float64)
        for i in range(Xn.shape[0]):
            w_row = weights[i].astype(np.float64, copy=False)
            s = float(np.sum(w_row))
            if s <= self.eps:
                scores[i] = 0.0
                continue
            w_norm = w_row / s
            pi_hat = float(np.dot(w_norm, self.pi_[inds[i]]))
            affinity = s / float(len(w_row))  # average weight (distance scale)
            scores[i] = -(pi_hat * affinity)

        return scores


@register_model(
    "vision_rgraph",
    tags=("vision", "classical", "rgraph", "graph"),
    metadata={"description": "Graph random-walk outlier detector (native, simplified)"},
)
class VisionRGraph(BaseVisionDetector):
    """Vision-friendly RGraph wrapper operating on extracted feature vectors."""

    def __init__(
        self,
        *,
        feature_extractor=None,
        contamination: float = 0.1,
        transition_steps: int = 10,
        n_nonzero: int = 10,
        gamma: float = 50.0,
        gamma_nz: bool = True,  # accepted for API compat (unused)
        algorithm: str = "lasso_lars",  # unused
        tau: float = 1.0,  # unused
        maxiter_lasso: int = 1000,  # unused
        preprocessing: bool = True,
        blocksize_test_data: int = 10,  # unused
        support_init: str = "L2",  # unused
        maxiter: int = 40,  # unused
        support_size: int = 100,  # unused
        active_support: bool = True,  # unused
        fit_intercept_LR: bool = False,  # unused
        verbose: bool = True,  # unused
        **kwargs,
    ) -> None:
        # Store all args for compatibility; the simplified core only uses a subset.
        self._detector_kwargs = dict(
            contamination=float(contamination),
            transition_steps=int(transition_steps),
            n_nonzero=int(n_nonzero),
            gamma=float(gamma),
            preprocessing=bool(preprocessing),
            gamma_nz=bool(gamma_nz),
            algorithm=str(algorithm),
            tau=float(tau),
            maxiter_lasso=int(maxiter_lasso),
            blocksize_test_data=int(blocksize_test_data),
            support_init=str(support_init),
            maxiter=int(maxiter),
            support_size=int(support_size),
            active_support=bool(active_support),
            fit_intercept_LR=bool(fit_intercept_LR),
            verbose=bool(verbose),
            **dict(kwargs),
        )
        super().__init__(contamination=contamination, feature_extractor=feature_extractor)

    def _build_detector(self):
        return CoreRGraph(**self._detector_kwargs)

    def fit(self, X: Iterable[str], y=None):
        return super().fit(X, y=y)

    def decision_function(self, X):
        return super().decision_function(X)

