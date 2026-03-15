# -*- coding: utf-8 -*-
"""
IForest (Isolation Forest) detector.

We use scikit-learn's `IsolationForest` and expose it via the unified `pyimgano`
vision API. The anomaly score is mapped to the `pyimgano` scoring convention:

- higher score => more anomalous
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.utils import check_array

from .baseml import BaseVisionDetector
from .core_feature_base import CoreFeatureDetector
from .registry import register_model


class CoreIForest:
    """Sklearn-backed Isolation Forest with sklearn-style scoring semantics."""

    def __init__(
        self,
        *,
        contamination: float = 0.1,
        n_estimators: int = 100,
        max_samples: Union[str, int] = "auto",
        max_features: float = 1.0,
        bootstrap: bool = False,
        n_jobs: int = 1,
        random_state: Optional[int] = None,
        verbose: int = 0,
        **kwargs,
    ) -> None:
        self.contamination = float(contamination)
        self._random_state = 0 if random_state is None else int(random_state)
        self._iforest_kwargs = dict(
            n_estimators=int(n_estimators),
            max_samples=max_samples,
            max_features=float(max_features),
            bootstrap=bool(bootstrap),
            n_jobs=int(n_jobs),
            verbose=int(verbose),
            **dict(kwargs),
        )

        self.iforest_: IsolationForest | None = None
        self.decision_scores_: np.ndarray | None = None

    def fit(self, x, y=None):  # noqa: ANN001, ANN201 - sklearn-like API
        del y
        x = check_array(x, ensure_2d=True, dtype=np.float64)
        # We compute thresholding ourselves via BaseDetector, so keep sklearn's
        # internal contamination handling out of the way.
        self.iforest_ = IsolationForest(
            contamination="auto",
            random_state=self._random_state,
            **self._iforest_kwargs,
        )
        self.iforest_.fit(x)
        self.decision_scores_ = self.decision_function(x)
        return self

    def decision_function(self, x):  # noqa: ANN001, ANN201 - sklearn-like API
        if self.iforest_ is None:
            raise RuntimeError("Detector must be fitted before calling decision_function")
        x = check_array(x, ensure_2d=True, dtype=np.float64)
        # sklearn: lower score => more abnormal. Invert to match `pyimgano` convention.
        return (-self.iforest_.score_samples(x)).astype(np.float64).ravel()


@register_model(
    "core_iforest",
    tags=("classical", "core", "features", "iforest", "ensemble", "baseline"),
    metadata={
        "description": "Core Isolation Forest on feature matrices (native wrapper)",
        "input": "features",
        "paper": "Liu et al., Isolation Forest (ICDM 2008)",
        "year": 2008,
    },
)
class CoreIForestModel(CoreFeatureDetector):
    """Core (feature-matrix) Isolation Forest detector with BaseDetector thresholding."""

    def __init__(
        self,
        *,
        contamination: float = 0.1,
        n_estimators: int = 100,
        max_samples: Union[str, int] = "auto",
        max_features: float = 1.0,
        bootstrap: bool = False,
        n_jobs: int = 1,
        random_state: Optional[int] = None,
        verbose: int = 0,
        **kwargs,
    ) -> None:
        self._backend_kwargs = dict(
            contamination=float(contamination),
            n_estimators=int(n_estimators),
            max_samples=max_samples,
            max_features=float(max_features),
            bootstrap=bool(bootstrap),
            n_jobs=int(n_jobs),
            random_state=random_state,
            verbose=int(verbose),
            **dict(kwargs),
        )
        super().__init__(contamination=float(contamination))

    def _build_detector(self):
        return CoreIForest(**self._backend_kwargs)


@register_model(
    "vision_iforest",
    tags=("vision", "classical", "iforest", "ensemble", "baseline"),
    metadata={
        "description": "Isolation Forest detector (baseline, robust general-purpose)",
        "paper": "Liu et al., Isolation Forest (ICDM 2008)",
        "year": 2008,
        "fast": True,
    },
)
class VisionIForest(BaseVisionDetector):
    """Vision-compatible Isolation Forest detector."""

    def __init__(
        self,
        *,
        feature_extractor=None,
        contamination: float = 0.1,
        n_estimators: int = 100,
        max_samples: Union[str, int] = "auto",
        max_features: float = 1.0,
        bootstrap: bool = False,
        n_jobs: int = 1,
        random_state: Optional[int] = None,
        verbose: int = 0,
        **kwargs,
    ) -> None:
        self._detector_kwargs = {
            "contamination": float(contamination),
            "n_estimators": int(n_estimators),
            "max_samples": max_samples,
            "max_features": float(max_features),
            "bootstrap": bool(bootstrap),
            "n_jobs": int(n_jobs),
            "random_state": random_state,
            "verbose": int(verbose),
            **dict(kwargs),
        }
        super().__init__(contamination=contamination, feature_extractor=feature_extractor)

    def _build_detector(self):
        return CoreIForest(**self._detector_kwargs)
