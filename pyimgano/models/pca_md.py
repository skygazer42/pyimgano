# -*- coding: utf-8 -*-
"""PCA + Mahalanobis distance detector.

Pipeline:
1) Fit PCA on training features
2) Compute Mahalanobis distance in the PCA subspace
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.decomposition import PCA
from sklearn.utils.validation import check_array

from ..utils.fitted import require_fitted
from .base_detector import BaseDetector
from .baseml import BaseVisionDetector
from .registry import register_model

logger = logging.getLogger(__name__)


@register_model(
    "core_pca_md",
    tags=("classical", "core", "features", "pca", "distance"),
    metadata={"description": "PCA + Mahalanobis distance (subspace)"},
)
class CorePCAMahalanobis(BaseDetector):
    def __init__(
        self,
        *,
        contamination: float = 0.1,
        n_components: int | float = 0.95,
        whiten: bool = False,
        reg: float = 1e-6,
        random_state: int | None = None,
    ) -> None:
        super().__init__(contamination=float(contamination))
        self.n_components = n_components
        self.whiten = bool(whiten)
        self.reg = float(reg)
        self.random_state = random_state

    def fit(self, x, y=None):  # noqa: ANN001, ANN201
        x_arr = check_array(x, ensure_2d=True, dtype=np.float64)
        self._set_n_classes(y)
        if x_arr.shape[0] == 0:
            raise ValueError("X must be non-empty")

        pca = PCA(
            n_components=self.n_components, whiten=self.whiten, random_state=self.random_state
        )
        z = pca.fit_transform(x_arr)

        mu = np.mean(z, axis=0)
        diff = z - mu
        cov = (diff.T @ diff) / max(1, z.shape[0] - 1)
        cov = cov + float(self.reg) * np.eye(cov.shape[0], dtype=np.float64)
        inv = np.linalg.pinv(cov)

        self.pca_ = pca
        self.mean_ = mu
        self.inv_cov_ = inv

        self.decision_scores_ = np.asarray(self.decision_function(x_arr), dtype=np.float64)
        self._process_decision_scores()
        return self

    def decision_function(self, x):  # noqa: ANN001, ANN201
        require_fitted(self, ["pca_", "mean_", "inv_cov_"])
        pca: PCA = self.pca_  # type: ignore[assignment]
        mu = np.asarray(self.mean_, dtype=np.float64)  # type: ignore[arg-type]
        inv = np.asarray(self.inv_cov_, dtype=np.float64)  # type: ignore[arg-type]

        x_arr = check_array(x, ensure_2d=True, dtype=np.float64)
        z = pca.transform(x_arr)
        diff = z - mu
        md2 = np.einsum("ij,jk,ik->i", diff, inv, diff)
        return np.asarray(md2, dtype=np.float64).reshape(-1)


@register_model(
    "vision_pca_md",
    tags=("vision", "classical", "pca", "distance"),
    metadata={"description": "Vision PCA + Mahalanobis distance (subspace)"},
)
class VisionPCAMahalanobis(BaseVisionDetector):
    def __init__(
        self,
        *,
        feature_extractor=None,
        contamination: float = 0.1,
        n_components: int | float = 0.95,
        whiten: bool = False,
        reg: float = 1e-6,
        random_state: int | None = None,
    ) -> None:
        self._detector_kwargs = {
            "contamination": float(contamination),
            "n_components": n_components,
            "whiten": bool(whiten),
            "reg": float(reg),
            "random_state": random_state,
        }
        logger.debug("Initializing VisionPCAMahalanobis with kwargs=%s", self._detector_kwargs)
        super().__init__(contamination=contamination, feature_extractor=feature_extractor)

    def _build_detector(self):
        return CorePCAMahalanobis(**self._detector_kwargs)
