# -*- coding: utf-8 -*-
"""
PCA (Principal Component Analysis) outlier detector.

PCA-based outlier detection uses reconstruction error from principal components
to identify anomalies. Samples with large reconstruction error are more likely
to be anomalous.

Reference:
    Shyu, M.L., Chen, S.C., Sarinnapakorn, K. and Chang, L., 2003.
    A novel anomaly detection scheme based on principal component classifier.
    ICDM.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_array

from .baseml import BaseVisionDetector
from .registry import register_model


class CorePCA:
    """Minimal PCA reconstruction-error detector (sklearn backend)."""

    def __init__(
        self,
        *,
        contamination: float = 0.1,
        n_components=None,
        n_selected_components: int | None = None,
        whiten: bool = False,
        svd_solver: str = "auto",
        weighted: bool = True,  # API compat (currently unused)
        standardization: bool = True,
        random_state=None,
        **pca_kwargs,
    ) -> None:
        self.contamination = float(contamination)
        self.n_components = n_components
        self.n_selected_components = n_selected_components
        self.whiten = bool(whiten)
        self.svd_solver = str(svd_solver)
        self.weighted = bool(weighted)
        self.standardization = bool(standardization)
        self.random_state = random_state
        self._pca_kwargs = dict(pca_kwargs)

        self.scaler_: StandardScaler | None = None
        self.pca_: PCA | None = None
        self.decision_scores_: np.ndarray | None = None

    def fit(self, X, y=None):  # noqa: ANN001, ANN201
        X = check_array(X, ensure_2d=True, dtype=np.float64)

        X_proc = X
        if self.standardization:
            self.scaler_ = StandardScaler()
            X_proc = self.scaler_.fit_transform(X_proc)

        self.pca_ = PCA(
            n_components=self.n_components,
            whiten=self.whiten,
            svd_solver=self.svd_solver,
            random_state=self.random_state,
            **self._pca_kwargs,
        )
        self.pca_.fit(X_proc)

        self.decision_scores_ = self.decision_function(X)
        return self

    def decision_function(self, X):  # noqa: ANN001, ANN201
        if self.pca_ is None:
            raise RuntimeError("Detector must be fitted before calling decision_function")

        X = check_array(X, ensure_2d=True, dtype=np.float64)
        X_proc = X
        if self.standardization:
            if self.scaler_ is None:
                raise RuntimeError("Internal error: missing scaler")
            X_proc = self.scaler_.transform(X_proc)

        Z = self.pca_.transform(X_proc)

        if self.n_selected_components is not None:
            k = int(self.n_selected_components)
            if k < 1:
                raise ValueError("n_selected_components must be >= 1")
            k = min(k, Z.shape[1])
            Z_masked = np.zeros_like(Z)
            Z_masked[:, :k] = Z[:, :k]
            X_recon = self.pca_.inverse_transform(Z_masked)
        else:
            X_recon = self.pca_.inverse_transform(Z)

        # Reconstruction error in the (optionally) standardized space.
        err = np.sum((X_proc - X_recon) ** 2, axis=1)
        return err.astype(np.float64).ravel()


@register_model(
    "vision_pca",
    tags=("vision", "classical", "linear", "pca"),
    metadata={
        "description": "Vision wrapper for PCA-based outlier detector",
        "paper": "ICDM 2003",
        "year": 2003,
        "classic": True,
        "interpretable": True,
    },
)
class VisionPCA(BaseVisionDetector):
    """Vision-compatible PCA detector for anomaly detection."""

    def __init__(
        self,
        *,
        feature_extractor=None,
        contamination: float = 0.1,
        n_components=None,
        n_selected_components: int | None = None,
        whiten: bool = False,
        svd_solver: str = "auto",
        weighted: bool = True,
        standardization: bool = True,
        random_state=None,
        **kwargs,
    ):
        self.detector_kwargs = dict(
            contamination=contamination,
            n_components=n_components,
            n_selected_components=n_selected_components,
            whiten=whiten,
            svd_solver=svd_solver,
            weighted=weighted,
            standardization=standardization,
            random_state=random_state,
            **kwargs,
        )
        super().__init__(contamination=contamination, feature_extractor=feature_extractor)

    def _build_detector(self):
        return CorePCA(**self.detector_kwargs)

    def fit(self, X: Iterable[str], y=None):
        return super().fit(X, y=y)

    def decision_function(self, X):
        return super().decision_function(X)

