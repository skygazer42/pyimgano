# -*- coding: utf-8 -*-
"""Elliptic envelope / robust covariance distance.

Industrial motivation
--------------------
For many factory problems, a surprisingly strong baseline is:

1) Extract embeddings (handcrafted or deep)
2) Fit a (robust) Gaussian model on normal embeddings
3) Score by Mahalanobis distance

This module provides a small, native wrapper around robust covariance fitting
using scikit-learn's covariance estimators, but exposed through the `pyimgano`
`BaseDetector` contract (thresholding + predict semantics).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.utils.validation import check_array

from pyimgano.utils.fitted import require_fitted

from .base_detector import BaseDetector
from .baseml import BaseVisionDetector
from .registry import register_model


def _fit_covariance(
    x: np.ndarray,
    *,
    robust: bool,
    support_fraction: Optional[float],
    random_state: Optional[int],
    assume_centered: bool,
):
    # Local import: keep `import pyimgano.models` light-ish.
    from sklearn.covariance import EmpiricalCovariance, MinCovDet

    if not robust:
        cov = EmpiricalCovariance(assume_centered=bool(assume_centered))
        cov.fit(x)
        return cov

    # Robust MCD can fail on degenerate/high-dim data. We fallback to empirical.
    try:
        cov = MinCovDet(
            support_fraction=support_fraction,
            random_state=random_state,
            assume_centered=bool(assume_centered),
        )
        cov.fit(x)
        return cov
    except Exception:
        cov = EmpiricalCovariance(assume_centered=bool(assume_centered))
        cov.fit(x)
        return cov


@register_model(
    "core_elliptic_envelope",
    tags=("classical", "core", "features", "gaussian", "covariance"),
    metadata={
        "description": "Robust covariance (MCD) Mahalanobis-distance outlier baseline",
        "type": "gaussian",
    },
)
class CoreEllipticEnvelope(BaseDetector):
    """Feature-matrix robust covariance baseline.

    Parameters
    ----------
    robust:
        If True, use Minimum Covariance Determinant (MCD) when possible.
        If False, fall back to empirical covariance.
    support_fraction:
        MCD support fraction (see scikit-learn). When None, sklearn chooses.
    assume_centered:
        If True, covariance is computed assuming data is centered at origin.
    """

    def __init__(
        self,
        *,
        contamination: float = 0.1,
        robust: bool = True,
        support_fraction: float | None = None,
        assume_centered: bool = False,
        random_state: int | None = None,
    ) -> None:
        super().__init__(contamination=float(contamination))
        self.robust = bool(robust)
        self.support_fraction = support_fraction
        self.assume_centered = bool(assume_centered)
        self.random_state = random_state

    def fit(self, x, y=None):  # noqa: ANN001, ANN201 - sklearn-like
        x_arr = check_array(x, ensure_2d=True, dtype=np.float64)
        self._set_n_classes(y)

        cov = _fit_covariance(
            x_arr,
            robust=bool(self.robust),
            support_fraction=self.support_fraction,
            random_state=self.random_state,
            assume_centered=bool(self.assume_centered),
        )

        # sklearn returns squared Mahalanobis distances (>= 0).
        scores = np.asarray(cov.mahalanobis(x_arr), dtype=np.float64).reshape(-1)

        self.covariance_ = cov
        self.decision_scores_ = scores
        self._process_decision_scores()
        return self

    def decision_function(self, x):  # noqa: ANN001, ANN201 - sklearn-like
        require_fitted(self, ["covariance_"])
        x_arr = check_array(x, ensure_2d=True, dtype=np.float64)
        cov = getattr(self, "covariance_")
        scores = np.asarray(cov.mahalanobis(x_arr), dtype=np.float64).reshape(-1)
        return scores


@register_model(
    "vision_elliptic_envelope",
    tags=("vision", "classical", "gaussian", "covariance"),
    metadata={
        "description": "Vision wrapper for robust covariance / Mahalanobis-distance baseline",
    },
)
class VisionEllipticEnvelope(BaseVisionDetector):
    def __init__(
        self,
        *,
        feature_extractor=None,
        contamination: float = 0.1,
        robust: bool = True,
        support_fraction: float | None = None,
        assume_centered: bool = False,
        random_state: int | None = None,
    ) -> None:
        self._detector_kwargs = {
            "contamination": float(contamination),
            "robust": bool(robust),
            "support_fraction": support_fraction,
            "assume_centered": bool(assume_centered),
            "random_state": random_state,
        }
        super().__init__(contamination=contamination, feature_extractor=feature_extractor)

    def _build_detector(self):
        return CoreEllipticEnvelope(**self._detector_kwargs)
