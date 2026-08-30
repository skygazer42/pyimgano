# -*- coding: utf-8 -*-
"""R-Graph outlier detection using sparse self-representation.

Each sample is represented by a sparse linear combination of the remaining
samples.  Absolute representation coefficients form a directed transition
matrix; low Cesaro-mean random-walk visitation identifies anomalies.  For
novel samples, a bounded query block is appended to the fitted reference set,
matching the transductive extension used by the established implementation.
"""

from __future__ import annotations

import warnings
from typing import Iterable

import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import sparse_encode
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_array

from ._legacy_x import MISSING
from .baseml import BaseVisionDetector
from .core_feature_base import CoreFeatureDetector
from .registry import register_model


class CoreRGraph:
    """Sparse self-representation and random-walk R-Graph core."""

    _legacy_attr_aliases = {"_train_X": "_train_x"}

    def __init__(
        self,
        *,
        contamination: float = 0.1,
        transition_steps: int = 10,
        n_nonzero: int = 10,
        gamma: float = 50.0,
        gamma_nz: bool = True,
        algorithm: str = "lasso_lars",
        tau: float = 1.0,
        maxiter_lasso: int = 1000,
        preprocessing: bool = True,
        blocksize_test_data: int = 1,
        support_init: str = "L2",
        maxiter: int = 40,
        support_size: int = 100,
        active_support: bool = True,
        fit_intercept_lr: bool = False,
        verbose: bool = False,
        # Accepted for compatibility with the former kNN proxy. They no longer
        # alter the paper algorithm and can be removed by callers.
        metric: str = "minkowski",
        p: int = 2,
        eps: float = 1e-12,
        **kwargs,
    ) -> None:
        legacy_fit_intercept = kwargs.pop("fit_intercept_LR", MISSING)
        if legacy_fit_intercept is not MISSING:
            if fit_intercept_lr:
                raise TypeError("CoreRGraph() got multiple values for argument 'fit_intercept_lr'")
            fit_intercept_lr = bool(legacy_fit_intercept)
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"Unknown RGraph parameters: {unknown}")

        self.contamination = float(contamination)
        self.transition_steps = int(transition_steps)
        self.n_nonzero = int(n_nonzero)
        self.gamma = float(gamma)
        self.gamma_nz = bool(gamma_nz)
        self.algorithm = str(algorithm)
        self.tau = float(tau)
        self.maxiter_lasso = int(maxiter_lasso)
        self.preprocessing = bool(preprocessing)
        self.blocksize_test_data = int(blocksize_test_data)
        self.support_init = str(support_init)
        self.maxiter = int(maxiter)
        self.support_size = int(support_size)
        self.active_support = bool(active_support)
        self.fit_intercept_lr = bool(fit_intercept_lr)
        self.verbose = bool(verbose)
        self.metric = str(metric)
        self.p = int(p)
        self.eps = float(eps)

        self.scaler_: StandardScaler | None = None
        self._train_x: NDArray[np.float64] | None = None
        self.representation_matrix_: NDArray[np.float64] | None = None
        self.transition_matrix_: NDArray[np.float64] | None = None
        self.pi_: NDArray[np.float64] | None = None
        self.decision_scores_: NDArray[np.float64] | None = None

    def __getattr__(self, name: str):
        alias = type(self)._legacy_attr_aliases.get(name)
        if alias is not None:
            return getattr(self, alias)
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")

    def __setattr__(self, name: str, value) -> None:
        alias = type(self)._legacy_attr_aliases.get(name)
        super().__setattr__(alias or name, value)

    def _validate_parameters(self) -> None:
        if self.transition_steps < 1:
            raise ValueError("transition_steps must be >= 1")
        if self.n_nonzero < 1:
            raise ValueError("n_nonzero must be >= 1")
        if self.gamma <= 0.0:
            raise ValueError("gamma must be > 0")
        if not 0.0 <= self.tau <= 1.0:
            raise ValueError("tau must be in [0, 1]")
        if self.algorithm not in {"lasso_lars", "lasso_cd"}:
            raise ValueError("algorithm must be 'lasso_lars' or 'lasso_cd'")
        if self.maxiter_lasso < 1 or self.maxiter < 1:
            raise ValueError("maxiter_lasso and maxiter must be >= 1")
        if self.blocksize_test_data < 1 or self.support_size < 1:
            raise ValueError("blocksize_test_data and support_size must be >= 1")
        if self.support_init.lower() not in {"l2", "knn"}:
            raise ValueError("support_init must be 'L2' or 'knn'")

    def _solve_sparse(
        self,
        dictionary: NDArray[np.float64],
        target: NDArray[np.float64],
        *,
        alpha: float,
    ) -> NDArray[np.float64]:
        if dictionary.shape[0] == 0:
            return np.zeros((0,), dtype=np.float64)
        if alpha <= self.eps or self.tau <= self.eps:
            model = LinearRegression(fit_intercept=self.fit_intercept_lr)
            model.fit(dictionary.T, target)
            return np.asarray(model.coef_, dtype=np.float64).reshape(-1)
        if self.tau >= 1.0 - 1e-10:
            with warnings.catch_warnings():
                # Degenerate atoms are common in self-representation dictionaries;
                # LARS drops them deterministically and still returns the solution.
                warnings.simplefilter("ignore", ConvergenceWarning)
                encoded = sparse_encode(
                    target[None, :],
                    dictionary,
                    algorithm=self.algorithm,
                    alpha=alpha,
                    max_iter=self.maxiter_lasso,
                )[0]
            return np.asarray(encoded, dtype=np.float64)

        # sklearn scales its residual by n_features; this conversion matches
        # 0.5*||y-cD||^2 + alpha*(tau*||c||_1 +(1-tau)/2*||c||_2^2).
        model = ElasticNet(
            alpha=alpha / float(max(1, target.shape[0])),
            l1_ratio=self.tau,
            fit_intercept=self.fit_intercept_lr,
            max_iter=self.maxiter_lasso,
            tol=1e-7,
            selection="cyclic",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(dictionary.T, target)
        return np.asarray(model.coef_, dtype=np.float64).reshape(-1)

    def _initial_support(
        self,
        dictionary: NDArray[np.float64],
        target: NDArray[np.float64],
        *,
        alpha: float,
        size: int,
    ) -> NDArray[np.int64]:
        if self.support_init.lower() == "knn":
            proxy = np.abs(dictionary @ target)
        else:
            gram = dictionary.T @ dictionary
            ridge = gram + max(alpha, self.eps) * np.eye(gram.shape[0])
            proxy = np.abs(dictionary @ np.linalg.solve(ridge, target))
        return np.argpartition(-proxy, size - 1)[:size].astype(np.int64, copy=False)

    def _active_support_coefficients(
        self,
        dictionary: NDArray[np.float64],
        target: NDArray[np.float64],
        *,
        alpha: float,
    ) -> NDArray[np.float64]:
        n_atoms = dictionary.shape[0]
        if not self.active_support or n_atoms <= self.support_size:
            return self._solve_sparse(dictionary, target, alpha=alpha)

        working_size = min(self.support_size, n_atoms)
        support = self._initial_support(dictionary, target, alpha=alpha, size=working_size)
        coefficients = np.zeros(n_atoms, dtype=np.float64)
        for _ in range(self.maxiter):
            local = self._solve_sparse(dictionary[support], target, alpha=alpha)
            coefficients.fill(0.0)
            coefficients[support] = local
            residual = target - coefficients @ dictionary
            coherence = np.abs(dictionary @ residual)
            coherence[support] = 0.0
            violating = np.flatnonzero(coherence > alpha * self.tau + 1e-10)
            if violating.size == 0:
                break
            active = np.flatnonzero(np.abs(coefficients) > 1e-10)
            room = max(0, working_size - active.size)
            if room == 0:
                support = active
                continue
            additions = violating[np.argsort(-coherence[violating])[:room]]
            support = np.unique(np.concatenate([active, additions])).astype(np.int64)
            if support.size == 0:
                break
        return coefficients

    def _self_representation(self, x_arr: NDArray[np.float64]) -> NDArray[np.float64]:
        n_samples = x_arr.shape[0]
        representation = np.zeros((n_samples, n_samples), dtype=np.float64)
        for index in range(n_samples):
            keep = np.arange(n_samples) != index
            dictionary = x_arr[keep]
            target = x_arr[index]
            if dictionary.shape[0] == 0:
                continue
            coherence = np.abs(dictionary @ target)
            if self.gamma_nz:
                alpha_zero = float(np.max(coherence, initial=0.0)) / max(self.tau, self.eps)
                alpha = alpha_zero / self.gamma
            else:
                alpha = 1.0 / self.gamma
            if self.gamma >= 1e4:
                alpha = 0.0
            coefficients = self._active_support_coefficients(dictionary, target, alpha=alpha)
            nonzero = np.flatnonzero(np.abs(coefficients) > 1e-10)
            if nonzero.size > self.n_nonzero:
                nonzero = nonzero[np.argsort(-np.abs(coefficients[nonzero]))[: self.n_nonzero]]
            original_indices = np.flatnonzero(keep)[nonzero]
            representation[index, original_indices] = coefficients[nonzero]
        return representation

    def _graph_scores(self, x_arr: NDArray[np.float64]) -> NDArray[np.float64]:
        # Sparse subspace representations are defined on direction rather than
        # magnitude. Unit-norm rows also prevent a large-magnitude query from
        # becoming an artificially cheap dictionary atom for every other row.
        norms = np.linalg.norm(x_arr, axis=1, keepdims=True)
        x_unit = np.divide(
            x_arr,
            norms,
            out=np.zeros_like(x_arr),
            where=norms > self.eps,
        )
        representation = self._self_representation(x_unit)
        transition = np.abs(representation)
        row_sums = np.sum(transition, axis=1, keepdims=True)
        transition = np.divide(
            transition,
            row_sums,
            out=np.zeros_like(transition),
            where=row_sums > self.eps,
        )

        n_samples = x_arr.shape[0]
        pi = np.full(n_samples, 1.0 / float(n_samples), dtype=np.float64)
        pi_bar = np.zeros(n_samples, dtype=np.float64)
        for _ in range(self.transition_steps):
            pi = pi @ transition
            pi_bar += pi
        pi_bar /= float(self.transition_steps)

        self.representation_matrix_ = representation
        self.transition_matrix_ = transition
        self.pi_ = pi_bar
        return -pi_bar

    def fit(self, x, y=None):  # noqa: ANN001, ANN201 - sklearn-like API
        del y
        x_arr = check_array(x, ensure_2d=True, dtype=np.float64)
        if x_arr.shape[0] == 0:
            raise ValueError("Training set cannot be empty")
        self._validate_parameters()

        if self.preprocessing:
            self.scaler_ = StandardScaler()
            x_norm = self.scaler_.fit_transform(x_arr)
        else:
            self.scaler_ = None
            x_norm = np.asarray(x_arr, dtype=np.float64)

        self.decision_scores_ = self._graph_scores(x_norm)
        self._train_x = x_norm.copy()
        return self

    def decision_function(self, x):  # noqa: ANN001, ANN201 - sklearn-like API
        if self._train_x is None:
            raise RuntimeError("Detector must be fitted before calling decision_function")
        x_arr = check_array(x, ensure_2d=True, dtype=np.float64)
        if x_arr.shape[1] != self._train_x.shape[1]:
            raise ValueError(f"Expected {self._train_x.shape[1]} features, got {x_arr.shape[1]}")
        if self.preprocessing and self.scaler_ is not None:
            x_norm = self.scaler_.transform(x_arr)
        else:
            x_norm = np.asarray(x_arr, dtype=np.float64)

        scores: list[NDArray[np.float64]] = []
        for start in range(0, x_norm.shape[0], self.blocksize_test_data):
            block = x_norm[start : start + self.blocksize_test_data]
            combined = np.vstack([self._train_x, block])
            scores.append(self._graph_scores(combined)[-block.shape[0] :])
        if not scores:
            return np.zeros((0,), dtype=np.float64)
        return np.concatenate(scores).astype(np.float64, copy=False)


@register_model(
    "core_rgraph",
    tags=("classical", "core", "features", "rgraph", "graph"),
    metadata={
        "description": "Sparse self-representation R-Graph with Cesaro random-walk scoring",
        "related_paper": "Provable Self-Representation Based Outlier Detection in a Union of Subspaces",
        "paper_url": "https://openaccess.thecvf.com/content_cvpr_2017/html/You_Provable_Self-Representation_Based_CVPR_2017_paper.html",
        "year": 2017,
        "paper_fidelity": "core-aligned",
        "implementation_status": "elastic-net-self-representation-random-walk",
        "known_deviation": "Novel samples are appended one at a time to prevent query-query contamination; metric and p remain no-op compatibility arguments from the retired kNN proxy.",
        "input": "features",
    },
)
class CoreRGraphModel(CoreFeatureDetector):
    """Core feature-matrix R-Graph detector with package thresholding."""

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
        self._backend_kwargs = dict(
            contamination=float(contamination),
            transition_steps=int(transition_steps),
            n_nonzero=int(n_nonzero),
            gamma=float(gamma),
            preprocessing=bool(preprocessing),
            metric=str(metric),
            p=int(p),
            eps=float(eps),
            **dict(kwargs),
        )
        super().__init__(contamination=float(contamination))

    def _build_detector(self):
        return CoreRGraph(**self._backend_kwargs)


@register_model(
    "vision_rgraph",
    tags=("vision", "classical", "rgraph", "graph"),
    metadata={
        "description": "Vision wrapper for sparse self-representation R-Graph",
        "related_paper": "Provable Self-Representation Based Outlier Detection in a Union of Subspaces",
        "paper_url": "https://openaccess.thecvf.com/content_cvpr_2017/html/You_Provable_Self-Representation_Based_CVPR_2017_paper.html",
        "year": 2017,
        "paper_fidelity": "core-aligned",
        "implementation_status": "vision-wrapper-over-self-representation-rgraph",
        "known_deviation": "Novel samples are appended one at a time to prevent query-query contamination.",
    },
)
class VisionRGraph(BaseVisionDetector):
    """Vision-friendly R-Graph wrapper operating on extracted features."""

    def __init__(
        self,
        *,
        feature_extractor=None,
        contamination: float = 0.1,
        transition_steps: int = 10,
        n_nonzero: int = 10,
        gamma: float = 50.0,
        gamma_nz: bool = True,
        algorithm: str = "lasso_lars",
        tau: float = 1.0,
        maxiter_lasso: int = 1000,
        preprocessing: bool = True,
        blocksize_test_data: int = 1,
        support_init: str = "L2",
        maxiter: int = 40,
        support_size: int = 100,
        active_support: bool = True,
        fit_intercept_lr: object = MISSING,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        legacy_fit_intercept_lr = kwargs.pop("fit_intercept_LR", MISSING)
        if fit_intercept_lr is MISSING:
            fit_intercept_lr_value = (
                False if legacy_fit_intercept_lr is MISSING else bool(legacy_fit_intercept_lr)
            )
        elif legacy_fit_intercept_lr is not MISSING:
            raise TypeError("VisionRGraph() got multiple values for argument 'fit_intercept_lr'")
        else:
            fit_intercept_lr_value = bool(fit_intercept_lr)

        self._detector_kwargs = dict(
            contamination=float(contamination),
            transition_steps=int(transition_steps),
            n_nonzero=int(n_nonzero),
            gamma=float(gamma),
            gamma_nz=bool(gamma_nz),
            algorithm=str(algorithm),
            tau=float(tau),
            maxiter_lasso=int(maxiter_lasso),
            preprocessing=bool(preprocessing),
            blocksize_test_data=int(blocksize_test_data),
            support_init=str(support_init),
            maxiter=int(maxiter),
            support_size=int(support_size),
            active_support=bool(active_support),
            fit_intercept_lr=fit_intercept_lr_value,
            verbose=bool(verbose),
            **dict(kwargs),
        )
        super().__init__(contamination=contamination, feature_extractor=feature_extractor)

    def _build_detector(self):
        return CoreRGraph(**self._detector_kwargs)

    def fit(self, x: Iterable[str], y=None):
        return super().fit(x, y=y)

    def decision_function(self, x):
        return super().decision_function(x)
