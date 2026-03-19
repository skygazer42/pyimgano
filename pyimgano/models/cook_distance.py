# -*- coding: utf-8 -*-
"""Cook's distance-inspired influence score (unsupervised baseline).

Classical Cook's distance is defined for supervised linear regression.
In industrial anomaly detection on embeddings, a similar intuition is useful:

- high **reconstruction residual** (doesn't fit normal subspace)
- high **leverage** (rare/atypical direction)

We implement a pragmatic unsupervised influence score:

1) Standardize features (z-score)
2) Fit PCA and compute reconstruction error per sample
3) Compute leverage from the (regularized) hat matrix diag
4) Score = residual^2 * leverage / (1 - leverage)^2

Higher score => more anomalous.
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


def _zscore(x: np.ndarray, *, eps: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = np.mean(x, axis=0)
    sd = np.std(x, axis=0)
    sd = np.where(sd > float(eps), sd, 1.0)
    z_scores = (x - mu) / sd
    return z_scores, mu, sd


def _leverage_diag(x: np.ndarray, *, ridge: float, eps: float) -> np.ndarray:
    # Hat matrix diag: diag(X (X^T X)^{-1} X^T)
    # Use ridge for stability on near-singular X^T X.
    d = int(x.shape[1])
    xtx = x.T @ x
    xtx = xtx + float(ridge) * np.eye(d, dtype=np.float64)
    inv = np.linalg.pinv(xtx)
    # diag(X @ inv @ X^T) computed without forming NxN:
    proj = x @ inv
    h = np.sum(proj * x, axis=1)
    h = np.clip(h, 0.0, 1.0 - float(eps))
    return np.asarray(h, dtype=np.float64).reshape(-1)


@register_model(
    "core_cook_distance",
    tags=("classical", "core", "features", "pca", "influence"),
    metadata={
        "description": "Cook-distance-inspired influence score (PCA residual + leverage)",
        "type": "influence",
    },
)
class CoreCookDistance(BaseDetector):
    def __init__(
        self,
        *,
        contamination: float = 0.1,
        n_components: int | float = 0.95,
        ridge: float = 1e-6,
        eps: float = 1e-12,
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__(contamination=float(contamination))
        self.n_components = n_components
        self.ridge = float(ridge)
        self.eps = float(eps)
        self.random_state = random_state

    def fit(self, x, y=None):  # noqa: ANN001, ANN201
        x_arr = check_array(x, ensure_2d=True, dtype=np.float64)
        self._set_n_classes(y)

        z_scores, mu, sd = _zscore(x_arr, eps=float(self.eps))
        pca = PCA(n_components=self.n_components, random_state=self.random_state)
        pca.fit(z_scores)

        scores = self._score_from_z(z_scores, pca)
        self._mu = mu
        self._sd = sd
        self._pca = pca
        self.decision_scores_ = scores
        self._process_decision_scores()
        return self

    def _score_from_z(self, z_scores: np.ndarray, pca: PCA) -> np.ndarray:
        # Reconstruction residual in standardized space.
        transformed = pca.transform(z_scores)
        z_hat = pca.inverse_transform(transformed)
        resid = np.linalg.norm(z_scores - z_hat, axis=1)

        h = _leverage_diag(z_scores, ridge=float(self.ridge), eps=float(self.eps))
        denom = np.maximum(np.square(1.0 - h), float(self.eps))
        score = (np.square(resid) * (h / denom)).astype(np.float64)
        return np.asarray(score, dtype=np.float64).reshape(-1)

    def decision_function(self, x):  # noqa: ANN001, ANN201
        require_fitted(self, ["_mu", "_sd", "_pca"])
        x_arr = check_array(x, ensure_2d=True, dtype=np.float64)
        mu = np.asarray(self._mu, dtype=np.float64)  # type: ignore[attr-defined]
        sd = np.asarray(self._sd, dtype=np.float64)  # type: ignore[attr-defined]
        sd = np.where(sd > float(self.eps), sd, 1.0)
        z_scores = (x_arr - mu) / sd
        return self._score_from_z(z_scores, self._pca)  # type: ignore[arg-type]


@register_model(
    "vision_cook_distance",
    tags=("vision", "classical", "pca", "influence"),
    metadata={"description": "Vision wrapper for Cook-distance-inspired influence score"},
)
class VisionCookDistance(BaseVisionDetector):
    def __init__(
        self,
        *,
        feature_extractor=None,
        contamination: float = 0.1,
        n_components: int | float = 0.95,
        ridge: float = 1e-6,
        eps: float = 1e-12,
        random_state: Optional[int] = None,
    ) -> None:
        self._detector_kwargs = {
            "contamination": float(contamination),
            "n_components": n_components,
            "ridge": float(ridge),
            "eps": float(eps),
            "random_state": random_state,
        }
        super().__init__(contamination=contamination, feature_extractor=feature_extractor)

    def _build_detector(self):
        return CoreCookDistance(**self._detector_kwargs)
