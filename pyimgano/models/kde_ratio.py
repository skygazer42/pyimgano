# -*- coding: utf-8 -*-
"""KDE ratio baseline (dual-bandwidth density contrast).

This is a pragmatic density-contrast score:
- Fit a **local** KDE (small bandwidth) capturing fine structure
- Fit a **global** KDE (large bandwidth) capturing coarse structure

Score = max(0, log p_global(x) - log p_local(x))

Intuition:
- Normal points in dense regions: local density >> global density => score ~ 0
- Isolated / boundary / anomalous points: local density not much higher => score > 0

Higher score => more anomalous.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_array

from pyimgano.utils.fitted import require_fitted

from .base_detector import BaseDetector
from .baseml import BaseVisionDetector
from .registry import register_model


@register_model(
    "core_kde_ratio",
    tags=("classical", "core", "features", "density", "kde"),
    metadata={
        "description": "Dual-bandwidth KDE density-contrast outlier score (native)",
        "type": "density",
    },
)
class CoreKDERatio(BaseDetector):
    def __init__(
        self,
        *,
        contamination: float = 0.1,
        bandwidth_local: float = 0.2,
        bandwidth_global: float = 1.0,
        kernel: str = "gaussian",
        standardize: bool = True,
    ) -> None:
        super().__init__(contamination=float(contamination))
        self.bandwidth_local = float(bandwidth_local)
        self.bandwidth_global = float(bandwidth_global)
        self.kernel = str(kernel)
        self.standardize = bool(standardize)

    def fit(self, X, y=None):  # noqa: ANN001, ANN201
        X_arr = check_array(X, ensure_2d=True, dtype=np.float64)
        self._set_n_classes(y)

        scaler = StandardScaler() if bool(self.standardize) else None
        Z = scaler.fit_transform(X_arr) if scaler is not None else X_arr

        kde_local = KernelDensity(bandwidth=float(self.bandwidth_local), kernel=str(self.kernel))
        kde_global = KernelDensity(bandwidth=float(self.bandwidth_global), kernel=str(self.kernel))
        kde_local.fit(Z)
        kde_global.fit(Z)

        self._scaler = scaler
        self._kde_local = kde_local
        self._kde_global = kde_global

        scores = self.decision_function(X_arr)
        self.decision_scores_ = np.asarray(scores, dtype=np.float64).reshape(-1)
        self._process_decision_scores()
        return self

    def decision_function(self, X):  # noqa: ANN001, ANN201
        require_fitted(self, ["_kde_local", "_kde_global"])
        X_arr = check_array(X, ensure_2d=True, dtype=np.float64)
        scaler = getattr(self, "_scaler", None)
        Z = scaler.transform(X_arr) if scaler is not None else X_arr

        ll_local = np.asarray(self._kde_local.score_samples(Z), dtype=np.float64).reshape(-1)  # type: ignore[attr-defined]
        ll_global = np.asarray(self._kde_global.score_samples(Z), dtype=np.float64).reshape(-1)  # type: ignore[attr-defined]
        score = ll_global - ll_local
        score = np.maximum(score, 0.0)
        return np.asarray(score, dtype=np.float64).reshape(-1)


@register_model(
    "vision_kde_ratio",
    tags=("vision", "classical", "density", "kde"),
    metadata={"description": "Vision wrapper for dual-bandwidth KDE density-contrast baseline"},
)
class VisionKDERatio(BaseVisionDetector):
    def __init__(
        self,
        *,
        feature_extractor=None,
        contamination: float = 0.1,
        bandwidth_local: float = 0.2,
        bandwidth_global: float = 1.0,
        kernel: str = "gaussian",
        standardize: bool = True,
    ) -> None:
        self._detector_kwargs = {
            "contamination": float(contamination),
            "bandwidth_local": float(bandwidth_local),
            "bandwidth_global": float(bandwidth_global),
            "kernel": str(kernel),
            "standardize": bool(standardize),
        }
        super().__init__(contamination=contamination, feature_extractor=feature_extractor)

    def _build_detector(self):
        return CoreKDERatio(**self._detector_kwargs)
