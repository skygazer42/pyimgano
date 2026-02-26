# -*- coding: utf-8 -*-
"""
ROD (Rotation-based Outlier Detector).

ROD is a robust, (largely) parameter-free method that operates naturally in 3D.
For higher dimensions, the original method aggregates scores across multiple 3D
subspaces.

Reference:
    Almardeny, Y. et al., 2020.
    A novel outlier detection approach based on Rodrigues rotation formula.

Pragmatic Note
--------------
The full nD variant that enumerates *all* 3D subspaces is combinatorial and can
be very slow when the feature dimension is large. To keep `pyimgano` fast and
lightweight, we cap the number of evaluated 3D subspaces by default.
"""

from __future__ import annotations

from itertools import combinations
from typing import Iterable, Sequence

import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.utils import check_array

from .baseml import BaseVisionDetector
from .registry import register_model


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _mad(costs: np.ndarray, median: float | None = None) -> tuple[np.ndarray, float]:
    costs = np.asarray(costs, dtype=np.float64).reshape(-1)
    med = float(np.nanmedian(costs)) if median is None else float(median)
    diff = np.abs(costs - med)
    denom = float(np.median(diff))
    if denom <= 0.0 or not np.isfinite(denom):
        return np.zeros_like(costs, dtype=np.float64), med
    z = 0.6745 * diff / denom
    return z.astype(np.float64), med


def _geometric_median(points: np.ndarray, eps: float = 1e-5, max_iter: int = 256) -> np.ndarray:
    """Weiszfeld's algorithm for geometric median."""

    pts = np.unique(np.asarray(points, dtype=np.float64), axis=0)
    if pts.shape[0] == 1:
        return pts[0]

    gm = np.mean(pts, axis=0)
    for _ in range(max_iter):
        diff = pts - gm
        dist = np.linalg.norm(diff, axis=1)
        nonzero = dist > 0
        if not np.any(nonzero):
            return gm

        inv = 1.0 / dist[nonzero]
        w = inv / np.sum(inv)
        t = np.sum(w[:, None] * pts[nonzero], axis=0)

        if np.linalg.norm(gm - t) < eps:
            return t
        gm = t
    return gm


def _fit_minmax_params(values: np.ndarray, *, feature_min: float, feature_max: float) -> tuple[float, float]:
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    if v.size == 0:
        # arbitrary but stable
        return float(feature_min), float(feature_max)
    data_min = float(min(np.min(v), feature_min))
    data_max = float(max(np.max(v), feature_max))
    return data_min, data_max


def _transform_minmax(values: np.ndarray, *, data_min: float, data_max: float, feature_min: float, feature_max: float) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    denom = data_max - data_min
    if denom <= 0.0 or not np.isfinite(denom):
        # Degenerate: map everything to the middle of the feature range.
        mid = (feature_min + feature_max) / 2.0
        return np.full_like(v, mid, dtype=np.float64)
    scaled = (v - data_min) / denom
    return scaled * (feature_max - feature_min) + feature_min


def _scale_angles(
    gammas: np.ndarray,
    *,
    params1: tuple[float, float] | None = None,
    params2: tuple[float, float] | None = None,
) -> tuple[np.ndarray, tuple[float, float], tuple[float, float]]:
    """Scale angles in two groups to stable ranges (train/test consistent)."""

    g = np.asarray(gammas, dtype=np.float64).reshape(-1)
    q1 = np.pi / 2.0

    first_mask = g <= q1
    second_mask = ~first_mask

    # Target ranges (borrowed from common ROD implementations)
    fmin, fmax = 0.001, 0.955
    smin, smax = q1 + 0.001, 2.186

    if params1 is None:
        params1 = _fit_minmax_params(g[first_mask], feature_min=fmin, feature_max=fmax)
    if params2 is None:
        params2 = _fit_minmax_params(g[second_mask], feature_min=smin, feature_max=smax)

    out = np.empty_like(g)
    if np.any(first_mask):
        out[first_mask] = _transform_minmax(
            g[first_mask],
            data_min=params1[0],
            data_max=params1[1],
            feature_min=fmin,
            feature_max=fmax,
        )
    if np.any(second_mask):
        out[second_mask] = _transform_minmax(
            g[second_mask],
            data_min=params2[0],
            data_max=params2[1],
            feature_min=smin,
            feature_max=smax,
        )
    return out, params1, params2


def _rod_3d(
    x: np.ndarray,
    *,
    gm: np.ndarray | None = None,
    mad_median: float | None = None,
    angle_params1: tuple[float, float] | None = None,
    angle_params2: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, float, tuple[float, float], tuple[float, float]]:
    """Compute ROD z-scores on 3D data and return fitted parameters."""

    x = np.asarray(x, dtype=np.float64)
    if x.shape[1] != 3:
        raise ValueError("rod_3d expects x to have shape (n_samples, 3)")

    gm_vec = _geometric_median(x) if gm is None else np.asarray(gm, dtype=np.float64)
    norm_gm = float(np.linalg.norm(gm_vec))

    centered = x - gm_vec
    v_norm = np.linalg.norm(centered, axis=1)
    denom = v_norm * max(norm_gm, 1e-12)
    cos_arg = np.divide(
        np.dot(centered, gm_vec),
        denom,
        out=np.zeros(x.shape[0], dtype=np.float64),
        where=denom > 0.0,
    )
    gammas = np.arccos(np.clip(cos_arg, -1.0, 1.0))
    gammas, angle_params1, angle_params2 = _scale_angles(
        gammas, params1=angle_params1, params2=angle_params2
    )

    costs = (v_norm**3) * np.cos(gammas) * (np.sin(gammas) ** 2)
    z, med = _mad(costs, median=mad_median)
    return z, gm_vec, med, angle_params1, angle_params2


def _choose_subspaces(
    dim: int,
    *,
    max_subspaces: int | None,
    random_state: int,
) -> list[tuple[int, int, int]]:
    total = dim * (dim - 1) * (dim - 2) // 6
    if max_subspaces is None or total <= max_subspaces:
        return list(combinations(range(dim), 3))

    rng = np.random.default_rng(random_state)
    chosen: set[tuple[int, int, int]] = set()
    attempts = 0
    target = int(max_subspaces)
    while len(chosen) < target and attempts < target * 50:
        idx = tuple(sorted(rng.choice(dim, size=3, replace=False).tolist()))
        chosen.add(idx)  # type: ignore[arg-type]
        attempts += 1
    return sorted(chosen)


class CoreROD:
    """Native ROD implementation with capped subspace evaluation."""

    def __init__(
        self,
        *,
        contamination: float = 0.1,
        parallel_execution: bool = False,  # API compat (currently ignored)
        max_subspaces: int | None = 256,
        random_state: int = 0,
    ) -> None:
        self.contamination = float(contamination)
        self.parallel_execution = bool(parallel_execution)
        self.max_subspaces = max_subspaces
        self.random_state = int(random_state)

        self._n_features_in: int | None = None
        self.data_scaler_: RobustScaler | None = None
        self.subspaces_: list[tuple[int, int, int]] | None = None

        # Per-subspace fitted params (n_subspaces items).
        self.gm_: list[np.ndarray] | None = None
        self.mad_median_: list[float] | None = None
        self.angle_params1_: list[tuple[float, float]] | None = None
        self.angle_params2_: list[tuple[float, float]] | None = None

        self.decision_scores_: np.ndarray | None = None

    def fit(self, X, y=None):  # noqa: ANN001, ANN201 - sklearn/pyod-like API
        X = check_array(X, ensure_2d=True, dtype=np.float64)
        self._n_features_in = int(X.shape[1])

        self.data_scaler_ = None
        self.subspaces_ = None
        self.gm_ = None
        self.mad_median_ = None
        self.angle_params1_ = None
        self.angle_params2_ = None

        self.decision_scores_ = self.decision_function(X)
        return self

    def decision_function(self, X):  # noqa: ANN001, ANN201 - sklearn/pyod-like API
        X = check_array(X, ensure_2d=True, dtype=np.float64)
        if self._n_features_in is None:
            raise RuntimeError("Detector must be fitted before calling decision_function")
        if X.shape[1] != self._n_features_in:
            raise ValueError(
                f"Expected {self._n_features_in} features, got {X.shape[1]}"
            )

        # Ensure at least 3 dims by padding zeros (ROD is 3D-native).
        if X.shape[1] < 3:
            pad = np.zeros((X.shape[0], 3 - X.shape[1]), dtype=np.float64)
            X_use = np.hstack([X, pad])
        else:
            X_use = X

        dim = X_use.shape[1]
        if dim == 3:
            # 3D: fit params once and reuse.
            if self.gm_ is None:
                z, gm, med, p1, p2 = _rod_3d(X_use)
                self.gm_ = [gm]
                self.mad_median_ = [med]
                self.angle_params1_ = [p1]
                self.angle_params2_ = [p2]
            else:
                z, _, _, _, _ = _rod_3d(
                    X_use,
                    gm=self.gm_[0],
                    mad_median=self.mad_median_[0],
                    angle_params1=self.angle_params1_[0],
                    angle_params2=self.angle_params2_[0],
                )
            return z.astype(np.float64).ravel()

        # nD: scale once, then score across (sampled) 3D subspaces.
        if self.data_scaler_ is None:
            self.data_scaler_ = RobustScaler()
            X_scaled = self.data_scaler_.fit_transform(X_use)
            self.subspaces_ = _choose_subspaces(
                dim, max_subspaces=self.max_subspaces, random_state=self.random_state
            )
            self.gm_, self.mad_median_, self.angle_params1_, self.angle_params2_ = [], [], [], []
        else:
            X_scaled = self.data_scaler_.transform(X_use)
            if self.subspaces_ is None:
                raise RuntimeError("Internal error: missing subspaces")
            if self.gm_ is None or self.mad_median_ is None or self.angle_params1_ is None or self.angle_params2_ is None:
                raise RuntimeError("Internal error: missing fitted parameters")

        assert self.subspaces_ is not None
        subspace_scores: list[np.ndarray] = []

        # Fit params per subspace during training; reuse on later calls.
        for idx, cols in enumerate(self.subspaces_):
            sub = X_scaled[:, cols]
            if self.gm_ is not None and idx < len(self.gm_):
                z, _, _, _, _ = _rod_3d(
                    sub,
                    gm=self.gm_[idx],
                    mad_median=self.mad_median_[idx],
                    angle_params1=self.angle_params1_[idx],
                    angle_params2=self.angle_params2_[idx],
                )
            else:
                z, gm, med, p1, p2 = _rod_3d(sub)
                self.gm_.append(gm)
                self.mad_median_.append(med)
                self.angle_params1_.append(p1)
                self.angle_params2_.append(p2)

            subspace_scores.append(_sigmoid(np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)))

        # Average "apples to apples" subspace scores.
        scores = np.mean(np.vstack(subspace_scores).T, axis=1)
        return scores.astype(np.float64).ravel()


@register_model(
    "vision_rod",
    tags=("vision", "classical", "rod", "baseline"),
    metadata={
        "description": "Rotation-based Outlier Detection (baseline)",
    },
)
class VisionROD(BaseVisionDetector):
    """Vision-compatible ROD detector."""

    def __init__(
        self,
        *,
        feature_extractor=None,
        contamination: float = 0.1,
        parallel_execution: bool = False,
        max_subspaces: int | None = 256,
        random_state: int = 0,
    ) -> None:
        self._detector_kwargs = {
            "contamination": float(contamination),
            "parallel_execution": bool(parallel_execution),
            "max_subspaces": max_subspaces,
            "random_state": int(random_state),
        }
        super().__init__(contamination=contamination, feature_extractor=feature_extractor)

    def _build_detector(self):
        return CoreROD(**self._detector_kwargs)

    def fit(self, X: Iterable[str], y=None):
        return super().fit(X, y=y)

    def decision_function(self, X):
        return super().decision_function(X)

