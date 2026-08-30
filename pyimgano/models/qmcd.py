"""
QMCD (Wrap-around Quasi-Monte Carlo Discrepancy) detector.

This detector uses wrap-around discrepancy as a robust-statistical feature and
converts it to a two-sided anomaly score around the training median.  The cited
uniform-design paper defines the discrepancy, not an anomaly detector or an
out-of-sample scoring rule.

Reference:
    Fang, K.T. and Ma, C.X., 2001. Wrap-Around L2-Discrepancy of Random
    Sampling, Latin Hypercube and Uniform Designs.
"""

# UPSTREAM: yzhao062/pyod @ 34f7996effac700a5166d882d5e94c6e6078fae3 (BSD-2-Clause; adapted)

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import check_array

from pyimgano.utils.optional_deps import require

from .baseml import BaseVisionDetector
from .core_feature_base import CoreFeatureDetector
from .registry import register_model

numba = require("numba", extra="numba", purpose="QMCD discrepancy acceleration")
njit = numba.njit


@njit(fastmath=True)
def _wrap_around_discrepancy(data: np.ndarray, check: np.ndarray) -> np.ndarray:
    """Wrap-around Quasi-Monte Carlo discrepancy.

    A serial JIT kernel is intentional: nested ``prange`` loops require a
    process-wide threading backend and caused runtime warnings or fallback on
    hosts with an older system TBB. The numerical kernel remains compiled and
    deterministic without that external threading-layer dependency.
    """

    n = data.shape[0]
    d = data.shape[1]
    p = check.shape[0]

    disc = np.zeros(p, dtype=np.float64)
    for i in range(p):
        dc = 0.0
        for j in range(n):
            prod = 1.0
            for k in range(d):
                x_kikj = abs(check[i, k] - data[j, k])
                prod *= 1.5 - x_kikj + x_kikj * x_kikj
            dc += prod
        disc[i] = dc

    return -((4.0 / 3.0) ** d) + (1.0 / (n**2)) * disc


class CoreQMCD:
    """Discrepancy-inspired detector with a robust two-sided score."""

    def __init__(self, *, contamination: float = 0.1, eps: float = 1e-12) -> None:
        self.contamination = float(contamination)
        self.eps = float(eps)

        self._scaler: MinMaxScaler | None = None
        self._fitted_data: np.ndarray | None = None
        self._score_center: float | None = None
        self._score_scale: float | None = None

        self.decision_scores_: np.ndarray | None = None

    def fit(self, x, _y=None):  # noqa: ANN001, ANN201 - sklearn-like API
        del _y
        x = check_array(x, ensure_2d=True, dtype=np.float64)

        self._scaler = MinMaxScaler()
        x_norm = self._scaler.fit_transform(x)
        self._fitted_data = x_norm.copy()

        raw_scores = np.asarray(_wrap_around_discrepancy(x_norm, x_norm), dtype=np.float64).ravel()
        self._score_center = float(np.median(raw_scores))
        mad = float(np.median(np.abs(raw_scores - self._score_center)))
        self._score_scale = max(1.4826 * mad, float(self.eps))
        self.decision_scores_ = np.abs(raw_scores - self._score_center) / self._score_scale
        return self

    def decision_function(self, x):  # noqa: ANN001, ANN201 - sklearn-like API
        if (
            self.decision_scores_ is None
            or self._scaler is None
            or self._fitted_data is None
            or self._score_center is None
            or self._score_scale is None
        ):
            raise RuntimeError("Detector must be fitted before calling decision_function")

        x = check_array(x, ensure_2d=True, dtype=np.float64)
        x_norm = self._scaler.transform(x)
        raw_scores = np.asarray(
            _wrap_around_discrepancy(self._fitted_data, x_norm), dtype=np.float64
        ).ravel()
        return np.abs(raw_scores - self._score_center) / self._score_scale


@register_model(
    "core_qmcd",
    tags=("classical", "core", "features", "qmcd", "robust", "baseline"),
    metadata={
        "description": "Wrap-around discrepancy-inspired robust two-sided detector",
        "input": "features",
        "related_paper": "Wrap-Around L2-Discrepancy of Random Sampling, Latin Hypercube and Uniform Designs",
        "paper_url": "https://doi.org/10.1006/jcom.2001.0589",
        "paper_fidelity": "inspired",
        "implementation_status": "discrepancy-inspired-robust-two-sided-score",
        "year": 2001,
    },
)
class CoreQMCDModel(CoreFeatureDetector):
    """Core (feature-matrix) QMCD detector with BaseDetector thresholding."""

    def __init__(self, *, contamination: float = 0.1, eps: float = 1e-12) -> None:
        self._backend_kwargs = {"contamination": float(contamination), "eps": float(eps)}
        super().__init__(contamination=float(contamination))

    def _build_detector(self):
        return CoreQMCD(**self._backend_kwargs)


@register_model(
    "vision_qmcd",
    tags=("vision", "classical", "qmcd", "robust", "baseline"),
    metadata={
        "description": "Wrap-around discrepancy-inspired robust-statistical baseline",
        "type": "robust-statistical",
        "related_paper": "Wrap-Around L2-Discrepancy of Random Sampling, Latin Hypercube and Uniform Designs",
        "paper_url": "https://doi.org/10.1006/jcom.2001.0589",
        "paper_fidelity": "inspired",
        "implementation_status": "discrepancy-inspired-robust-two-sided-score",
    },
)
class VisionQMCD(BaseVisionDetector):
    """Vision-compatible QMCD detector."""

    def __init__(
        self,
        *,
        feature_extractor=None,
        contamination: float = 0.1,
        eps: float = 1e-12,
    ) -> None:
        self._detector_kwargs = {"contamination": float(contamination), "eps": float(eps)}
        super().__init__(contamination=contamination, feature_extractor=feature_extractor)

    def _build_detector(self):
        return CoreQMCD(**self._detector_kwargs)

    def fit(self, x: Iterable[str], y=None):
        return super().fit(x, y=y)

    def decision_function(self, x):
        return super().decision_function(x)
