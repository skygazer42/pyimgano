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
from typing import ClassVar

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
    _legacy_attr_aliases: ClassVar[dict[str, str]] = {
        "row_mean_A": "row_mean_a",
        "row_mean_B": "row_mean_b",
        "grand_mean_A": "grand_mean_a",
        "grand_mean_B": "grand_mean_b",
    }

    w_u: np.ndarray
    w_v: np.ndarray
    u_train: np.ndarray
    v_train: np.ndarray
    row_mean_a: np.ndarray
    row_mean_b: np.ndarray
    grand_mean_a: float
    grand_mean_b: float

    def __getattr__(self, name: str):
        alias = type(self)._legacy_attr_aliases.get(name)
        if alias is not None:
            return getattr(self, alias)
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")


def _train_centering_stats(z: np.ndarray) -> tuple[np.ndarray, float]:
    """Return (row_means, grand_mean) for the |z_i - z_j| distance matrix."""

    z = np.asarray(z, dtype=np.float64).reshape(-1)
    d = np.abs(z[:, None] - z[None, :])
    row_mean = np.mean(d, axis=1)
    grand_mean = float(np.mean(d))
    return row_mean, grand_mean


def _train_contrib(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Per-sample contribution to distance covariance for training data."""

    u = np.asarray(u, dtype=np.float64).reshape(-1)
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    a = np.abs(u[:, None] - u[None, :])
    b = np.abs(v[:, None] - v[None, :])

    row_a = np.mean(a, axis=1)
    row_b = np.mean(b, axis=1)
    grand_a = float(np.mean(a))
    grand_b = float(np.mean(b))

    a_tilde = a - row_a[:, None] - row_a[None, :] + grand_a
    b_tilde = b - row_b[:, None] - row_b[None, :] + grand_b

    contrib = np.mean(a_tilde * b_tilde, axis=1)
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

    a = np.abs(u_test[:, None] - state.u_train[None, :])
    b = np.abs(v_test[:, None] - state.v_train[None, :])

    row_mean_a_test = np.mean(a, axis=1)
    row_mean_b_test = np.mean(b, axis=1)

    a_tilde = a - row_mean_a_test[:, None] - state.row_mean_a[None, :] + float(state.grand_mean_a)
    b_tilde = b - row_mean_b_test[:, None] - state.row_mean_b[None, :] + float(state.grand_mean_b)

    contrib = np.mean(a_tilde * b_tilde, axis=1)
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
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        super().__init__(contamination=float(contamination))
        self.n_projections = int(n_projections)
        self.random_state = random_state

    def fit(self, x, y=None):  # noqa: ANN001, ANN201
        x_arr = check_array(x, ensure_2d=True, dtype=np.float64)
        self._set_n_classes(y)
        n, d = x_arr.shape
        if n == 0:
            raise ValueError("x must be non-empty")
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

            u = x_arr @ w_u
            v = x_arr @ w_v

            row_a, grand_a = _train_centering_stats(u)
            row_b, grand_b = _train_centering_stats(v)
            score_proj = _train_contrib(u, v)

            states.append(
                _ProjState(
                    w_u=w_u,
                    w_v=w_v,
                    u_train=np.asarray(u, dtype=np.float64),
                    v_train=np.asarray(v, dtype=np.float64),
                    row_mean_a=np.asarray(row_a, dtype=np.float64),
                    row_mean_b=np.asarray(row_b, dtype=np.float64),
                    grand_mean_a=float(grand_a),
                    grand_mean_b=float(grand_b),
                )
            )
            scores += score_proj

        scores /= float(len(states))
        self._states = states
        self.decision_scores_ = scores
        self._process_decision_scores()
        return self

    def decision_function(self, x):  # noqa: ANN001, ANN201
        require_fitted(self, ["_states"])
        states: list[_ProjState] = self._states  # type: ignore[assignment]

        x_arr = check_array(x, ensure_2d=True, dtype=np.float64)
        if x_arr.shape[0] == 0:
            return np.zeros((0,), dtype=np.float64)

        scores = np.zeros((x_arr.shape[0],), dtype=np.float64)
        for st in states:
            u_test = x_arr @ st.w_u
            v_test = x_arr @ st.w_v
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
        random_state: int | np.random.Generator | None = None,
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
