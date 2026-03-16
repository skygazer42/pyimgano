# -*- coding: utf-8 -*-
"""CrossMAD (CVPR 2025) — pragmatic prototype-distance implementation.

Important note
--------------
The original CrossMAD paper proposes cross-modal prototype harmonization. This
repository implements a **dependency-stable industrial approximation** that
fits naturally into our "deep embeddings + classical core" architecture:

  images -> embedding extractor -> K prototypes -> distance-to-prototypes score

This yields a strong and simple baseline for industrial inspection while:
- avoiding runtime dependency on external outlier toolboxes
- avoiding implicit model weight downloads by default
- conforming to `BaseDetector` semantics (higher score => more anomalous)

We expose two registry models:
- `core_crossmad`: feature-matrix detector (N,D)
- `vision_crossmad`: vision wrapper using feature extractor registry
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.utils import check_array

from .baseml import BaseVisionDetector
from .core_feature_base import CoreFeatureDetector
from .registry import register_model


def _l2_normalize_rows(X: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    x_float = np.asarray(X, dtype=np.float64)
    denom = np.linalg.norm(x_float, axis=1, keepdims=True)
    denom = np.maximum(denom, float(eps))
    return x_float / denom


class CoreCrossMAD:
    """Prototype-distance detector (KMeans centroids + min distance).

    Parameters
    ----------
    num_prototypes:
        Number of prototypes (KMeans clusters). Automatically clamped to
        `<= n_samples` at fit time.
    metric:
        Distance metric for scoring, passed to `sklearn.metrics.pairwise_distances`.
        Common choices: 'euclidean', 'cosine'.
    normalize:
        If True, L2-normalize feature rows before clustering/scoring (recommended
        when using cosine or deep embeddings).
    """

    def __init__(
        self,
        *,
        contamination: float = 0.1,  # kept for signature compatibility
        num_prototypes: int = 10,
        metric: str = "euclidean",
        normalize: bool = True,
        random_state: Optional[int] = None,
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-4,
        algorithm: str = "lloyd",
        **kmeans_kwargs,
    ) -> None:
        self.contamination = float(contamination)
        self.num_prototypes = int(num_prototypes)
        self.metric = str(metric)
        self.normalize = bool(normalize)
        self.random_state = random_state
        self.n_init = int(n_init)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.algorithm = str(algorithm)
        self._kmeans_kwargs = dict(kmeans_kwargs)

        self.centers_: np.ndarray | None = None
        self.decision_scores_: np.ndarray | None = None

    def fit(self, X, y=None):  # noqa: ANN001, ANN201 - sklearn-like API
        _ = y
        x_np = check_array(X, ensure_2d=True, dtype=np.float64)
        n = int(x_np.shape[0])
        if n == 0:
            raise ValueError("Training set cannot be empty")

        if self.normalize:
            x_fit = _l2_normalize_rows(x_np)
        else:
            x_fit = x_np

        k = int(self.num_prototypes)
        if k < 1:
            raise ValueError("num_prototypes must be >= 1")
        k = min(k, n)

        km = KMeans(
            n_clusters=int(k),
            random_state=self.random_state,
            n_init=int(self.n_init),
            max_iter=int(self.max_iter),
            tol=float(self.tol),
            algorithm=str(self.algorithm),
            **self._kmeans_kwargs,
        )
        km.fit(x_fit)

        self.centers_ = np.asarray(km.cluster_centers_, dtype=np.float64)
        self.decision_scores_ = np.asarray(self.decision_function(x_np), dtype=np.float64).reshape(
            -1
        )
        return self

    def decision_function(self, X):  # noqa: ANN001, ANN201 - sklearn-like API
        if self.centers_ is None:
            raise RuntimeError("Detector must be fitted before calling decision_function")

        x_np = check_array(X, ensure_2d=True, dtype=np.float64)
        x_eval = _l2_normalize_rows(x_np) if self.normalize else x_np

        d = pairwise_distances(x_eval, self.centers_, metric=str(self.metric))
        return np.min(d, axis=1).astype(np.float64, copy=False).reshape(-1)


@register_model(
    "core_crossmad",
    tags=("classical", "core", "features", "crossmad", "prototype", "clustering"),
    metadata={
        "description": "Core CrossMAD-style prototype-distance detector on feature matrices (native)",
        "paper": "Beyond Single-Modal Boundary: Cross-Modal Anomaly Detection through Visual Prototype and Harmonization",
        "year": 2025,
        "conference": "CVPR",
        "input": "features",
    },
)
class CoreCrossMADModel(CoreFeatureDetector):
    def __init__(
        self,
        *,
        contamination: float = 0.1,
        num_prototypes: int = 10,
        metric: str = "euclidean",
        normalize: bool = True,
        random_state: Optional[int] = None,
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-4,
        algorithm: str = "lloyd",
        **kmeans_kwargs,
    ) -> None:
        self._backend_kwargs = dict(
            contamination=float(contamination),
            num_prototypes=int(num_prototypes),
            metric=str(metric),
            normalize=bool(normalize),
            random_state=random_state,
            n_init=int(n_init),
            max_iter=int(max_iter),
            tol=float(tol),
            algorithm=str(algorithm),
            **dict(kmeans_kwargs),
        )
        super().__init__(contamination=float(contamination))

    def _build_detector(self):
        return CoreCrossMAD(**self._backend_kwargs)


@register_model(
    "vision_crossmad",
    tags=("vision", "classical", "crossmad", "prototype", "embeddings", "cvpr2025"),
    metadata={
        "description": "Vision CrossMAD-style prototype-distance detector (embeddings + core_crossmad)",
        "paper": "Beyond Single-Modal Boundary: Cross-Modal Anomaly Detection through Visual Prototype and Harmonization",
        "year": 2025,
        "conference": "CVPR",
    },
    overwrite=True,
)
class VisionCrossMAD(BaseVisionDetector):
    """Vision wrapper around :class:`CoreCrossMAD`.

    Default feature extractor is a safe torchvision embedding extractor with
    `pretrained=False` to avoid implicit weight downloads.
    """

    def __init__(
        self,
        *,
        contamination: float = 0.1,
        feature_extractor=None,
        backbone: str = "resnet18",
        pretrained: bool = False,
        device: str = "cpu",
        image_size: int = 224,
        num_prototypes: int = 10,
        metric: str = "euclidean",
        normalize: bool = True,
        random_state: Optional[int] = None,
        prototype_dim: int | None = None,  # legacy compat (unused)
        **kwargs,
    ) -> None:
        # Legacy compat: earlier experimental versions used "wide_resnet50".
        bb = str(backbone)
        if bb == "wide_resnet50":
            bb = "wide_resnet50_2"

        if feature_extractor is None:
            feature_extractor = {
                "name": "torchvision_backbone",
                "kwargs": {
                    "backbone": str(bb),
                    "pretrained": bool(pretrained),
                    "device": str(device),
                    "image_size": int(image_size),
                },
            }

        # Keep for introspection / backward compatibility, but do not enforce.
        self.prototype_dim = prototype_dim

        self._detector_kwargs = dict(
            contamination=float(contamination),
            num_prototypes=int(num_prototypes),
            metric=str(metric),
            normalize=bool(normalize),
            random_state=random_state,
            **dict(kwargs),
        )
        super().__init__(contamination=float(contamination), feature_extractor=feature_extractor)

    def _build_detector(self):
        return CoreCrossMAD(**self._detector_kwargs)

    def fit(self, X: Iterable, y=None):  # noqa: ANN001, ANN201
        return super().fit(X, y=y)

    def decision_function(self, X):  # noqa: ANN001, ANN201
        return super().decision_function(X)


__all__ = ["CoreCrossMADModel", "VisionCrossMAD"]
