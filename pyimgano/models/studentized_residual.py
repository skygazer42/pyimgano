# -*- coding: utf-8 -*-
"""Studentized residual baseline (unsupervised).

We compute a reconstruction residual (PCA) and standardize it using a robust
scale estimate (median + MAD). This provides a stable 1D anomaly score that is
less sensitive to feature scaling and outliers in the training set.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.utils.validation import check_array

from pyimgano.utils.fitted import require_fitted

from .base_detector import BaseDetector
from .baseml import BaseVisionDetector
from .registry import register_model


def _mad_scale(x: np.ndarray, *, eps: float) -> tuple[float, float]:
    v = np.asarray(x, dtype=np.float64).reshape(-1)
    if v.size == 0:
        return 0.0, 1.0
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    mad = max(mad, float(eps))
    return med, mad


@register_model(
    "core_studentized_residual",
    tags=("classical", "core", "features", "pca", "residual"),
    metadata={
        "description": "PCA reconstruction residual standardized by median+MAD (studentized residual baseline)",
        "type": "residual",
    },
)
class CoreStudentizedResidual(BaseDetector):
    def __init__(
        self,
        *,
        contamination: float = 0.1,
        n_components: int | float = 0.95,
        eps: float = 1e-12,
        clamp_zero: bool = True,
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__(contamination=float(contamination))
        self.n_components = n_components
        self.eps = float(eps)
        self.clamp_zero = bool(clamp_zero)
        self.random_state = random_state

    def fit(self, x, y=None):  # noqa: ANN001, ANN201
        x_arr = check_array(x, ensure_2d=True, dtype=np.float64)
        self._set_n_classes(y)

        mu = np.mean(x_arr, axis=0)
        sd = np.std(x_arr, axis=0)
        sd = np.where(sd > float(self.eps), sd, 1.0)
        z = (x_arr - mu) / sd

        pca = PCA(n_components=self.n_components, random_state=self.random_state)
        pca.fit(z)

        resid = self._residual(z, pca)
        med, mad = _mad_scale(resid, eps=float(self.eps))

        self._mu = mu
        self._sd = sd
        self._pca = pca
        self._resid_median = med
        self._resid_mad = mad

        scores = self._standardize(resid)
        self.decision_scores_ = scores
        self._process_decision_scores()
        return self

    def _residual(self, z: np.ndarray, pca: PCA) -> np.ndarray:
        transformed = pca.transform(z)
        z_hat = pca.inverse_transform(transformed)
        resid = np.linalg.norm(z - z_hat, axis=1)
        return np.asarray(resid, dtype=np.float64).reshape(-1)

    def _standardize(self, resid: np.ndarray) -> np.ndarray:
        r = np.asarray(resid, dtype=np.float64).reshape(-1)
        med = float(self._resid_median)  # type: ignore[attr-defined]
        mad = float(self._resid_mad)  # type: ignore[attr-defined]
        z = (r - med) / max(mad, float(self.eps))
        if bool(self.clamp_zero):
            z = np.maximum(z, 0.0)
        return np.asarray(z, dtype=np.float64).reshape(-1)

    def decision_function(self, x):  # noqa: ANN001, ANN201
        require_fitted(self, ["_mu", "_sd", "_pca", "_resid_median", "_resid_mad"])
        x_arr = check_array(x, ensure_2d=True, dtype=np.float64)
        mu = np.asarray(self._mu, dtype=np.float64)  # type: ignore[attr-defined]
        sd = np.asarray(self._sd, dtype=np.float64)  # type: ignore[attr-defined]
        sd = np.where(sd > float(self.eps), sd, 1.0)
        z = (x_arr - mu) / sd
        resid = self._residual(z, self._pca)  # type: ignore[arg-type]
        return self._standardize(resid)


@register_model(
    "vision_studentized_residual",
    tags=("vision", "classical", "pca", "residual"),
    metadata={"description": "Vision wrapper for studentized PCA residual baseline"},
)
class VisionStudentizedResidual(BaseVisionDetector):
    def __init__(
        self,
        *,
        feature_extractor=None,
        contamination: float = 0.1,
        n_components: int | float = 0.95,
        eps: float = 1e-12,
        clamp_zero: bool = True,
        random_state: Optional[int] = None,
    ) -> None:
        self._detector_kwargs = {
            "contamination": float(contamination),
            "n_components": n_components,
            "eps": float(eps),
            "clamp_zero": bool(clamp_zero),
            "random_state": random_state,
        }
        super().__init__(contamination=contamination, feature_extractor=feature_extractor)

    def _build_detector(self):
        return CoreStudentizedResidual(**self._detector_kwargs)
