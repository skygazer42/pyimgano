# -*- coding: utf-8 -*-
"""Distance-correlation influence detector (lightweight).

Distance correlation is a dependence measure defined via pairwise distances.
Classic distance correlation is a *global* statistic; here we adapt it into a
per-sample unsupervised anomaly score by measuring each sample's contribution
to the distance-covariance term under random 1D projections.

This is intended as a niche classical detector for embedding/tabular features.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.utils.validation import check_array

from ..utils.fitted import require_fitted
from ..utils.random_state import check_random_state
from .base_detector import BaseDetector
from .baseml import BaseVisionDetector
from .registry import register_model

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _ProjState:
    w_u: np.ndarray
    w_v: np.ndarray
    u_train: np.ndarray
    v_train: np.ndarray
    row_mean_A: np.ndarray
    row_mean_B: np.ndarray
    grand_mean_A: float
    grand_mean_B: float


def _train_centering_stats(z: np.ndarray) -> tuple[np.ndarray, float]:
    """Return (row_means, grand_mean) for the |z_i - z_j| distance matrix."""

    z = np.asarray(z, dtype=np.float64).reshape(-1)
    D = np.abs(z[:, None] - z[None, :])
    row_mean = np.mean(D, axis=1)
    grand_mean = float(np.mean(D))
    return row_mean, grand_mean


def _train_contrib(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Per-sample contribution to distance covariance for training data."""

    u = np.asarray(u, dtype=np.float64).reshape(-1)
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    A = np.abs(u[:, None] - u[None, :])
    B = np.abs(v[:, None] - v[None, :])

    rowA = np.mean(A, axis=1)
    rowB = np.mean(B, axis=1)
    grandA = float(np.mean(A))
    grandB = float(np.mean(B))

    Atilde = A - rowA[:, None] - rowA[None, :] + grandA
    Btilde = B - rowB[:, None] - rowB[None, :] + grandB

    contrib = np.mean(Atilde * Btilde, axis=1)
    return np.abs(contrib)


def _test_contrib(
    *,
    u_test: np.ndarray,
    v_test: np.ndarray,
    state: _ProjState,
) -> np.ndarray:
    """Contribution score for test samples using training centering stats."""

    u_test = np.asarray(u_test, dtype=np.float64).reshape(-1)
    v_test = np.asarray(v_test, dtype=np.float64).reshape(-1)

    A = np.abs(u_test[:, None] - state.u_train[None, :])
    B = np.abs(v_test[:, None] - state.v_train[None, :])

    row_mean_A_test = np.mean(A, axis=1)
    row_mean_B_test = np.mean(B, axis=1)

    Atilde = A - row_mean_A_test[:, None] - state.row_mean_A[None, :] + float(state.grand_mean_A)
    Btilde = B - row_mean_B_test[:, None] - state.row_mean_B[None, :] + float(state.grand_mean_B)

    contrib = np.mean(Atilde * Btilde, axis=1)
    return np.abs(contrib)


@register_model(
    "core_dcorr",
    tags=("classical", "core", "features", "projection", "dependence"),
    metadata={"description": "Distance-correlation influence (random projections; native)"},
)
class CoreDCorr(BaseDetector):
    def __init__(
        self,
        *,
        contamination: float = 0.1,
        n_projections: int = 8,
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        super().__init__(contamination=float(contamination))
        self.n_projections = int(n_projections)
        self.random_state = random_state

    def fit(self, X, y=None):  # noqa: ANN001, ANN201
        X_arr = check_array(X, ensure_2d=True, dtype=np.float64)
        self._set_n_classes(y)
        n, d = X_arr.shape
        if n == 0:
            raise ValueError("X must be non-empty")
        if self.n_projections <= 0:
            raise ValueError("n_projections must be > 0")

        rng = check_random_state(self.random_state)
        states: list[_ProjState] = []
        scores = np.zeros((n,), dtype=np.float64)

        for _ in range(int(self.n_projections)):
            w_u = rng.normal(size=(d,)).astype(np.float64)
            w_v = rng.normal(size=(d,)).astype(np.float64)
            # Normalize to keep projection scale roughly stable.
            w_u /= max(float(np.linalg.norm(w_u)), 1e-12)
            w_v /= max(float(np.linalg.norm(w_v)), 1e-12)

            u = X_arr @ w_u
            v = X_arr @ w_v

            rowA, grandA = _train_centering_stats(u)
            rowB, grandB = _train_centering_stats(v)
            score_proj = _train_contrib(u, v)

            states.append(
                _ProjState(
                    w_u=w_u,
                    w_v=w_v,
                    u_train=np.asarray(u, dtype=np.float64),
                    v_train=np.asarray(v, dtype=np.float64),
                    row_mean_A=np.asarray(rowA, dtype=np.float64),
                    row_mean_B=np.asarray(rowB, dtype=np.float64),
                    grand_mean_A=float(grandA),
                    grand_mean_B=float(grandB),
                )
            )
            scores += score_proj

        scores /= float(len(states))
        self._states = states
        self.decision_scores_ = scores
        self._process_decision_scores()
        return self

    def decision_function(self, X):  # noqa: ANN001, ANN201
        require_fitted(self, ["_states"])
        states: list[_ProjState] = self._states  # type: ignore[assignment]

        X_arr = check_array(X, ensure_2d=True, dtype=np.float64)
        if X_arr.shape[0] == 0:
            return np.zeros((0,), dtype=np.float64)

        scores = np.zeros((X_arr.shape[0],), dtype=np.float64)
        for st in states:
            u_test = X_arr @ st.w_u
            v_test = X_arr @ st.w_v
            scores += _test_contrib(u_test=u_test, v_test=v_test, state=st)

        scores /= float(len(states))
        return scores


@register_model(
    "vision_dcorr",
    tags=("vision", "classical", "projection", "dependence"),
    metadata={"description": "Vision distance-correlation influence detector"},
)
class VisionDCorr(BaseVisionDetector):
    def __init__(
        self,
        *,
        feature_extractor=None,
        contamination: float = 0.1,
        n_projections: int = 8,
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        self._detector_kwargs = {
            "contamination": float(contamination),
            "n_projections": int(n_projections),
            "random_state": random_state,
        }
        logger.debug("Initializing VisionDCorr with kwargs=%s", self._detector_kwargs)
        super().__init__(contamination=contamination, feature_extractor=feature_extractor)

    def _build_detector(self):
        return CoreDCorr(**self._detector_kwargs)
