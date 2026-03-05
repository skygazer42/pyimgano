# -*- coding: utf-8 -*-
"""PatchCore-lite (coreset memory bank on embeddings).

This is an industrially useful "lite" approximation inspired by PatchCore:
- extract an embedding per image
- build a memory bank of normal embeddings (optionally subsampled / "coreset")
- score by nearest-neighbor distance to memory bank

Unlike full PatchCore, this implementation is **image-level only** (no pixel map),
so it stays lightweight and integrates well with our classical pipeline stack.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_array

from pyimgano.utils.fitted import require_fitted

from .base_detector import BaseDetector
from .baseml import BaseVisionDetector
from .registry import register_model


@register_model(
    "core_patchcore_lite",
    tags=("classical", "core", "features", "neighbors", "memory_bank", "patchcore"),
    metadata={
        "description": "PatchCore-lite: coreset memory bank + nearest-neighbor distance (image-level)",
        "type": "neighbors",
    },
)
class CorePatchCoreLite(BaseDetector):
    def __init__(
        self,
        *,
        contamination: float = 0.1,
        coreset_ratio: float = 0.2,
        metric: str = "minkowski",
        p: int = 2,
        random_state: Optional[int] = 0,
    ) -> None:
        super().__init__(contamination=float(contamination))
        self.coreset_ratio = float(coreset_ratio)
        self.metric = str(metric)
        self.p = int(p)
        self.random_state = random_state

    def fit(self, X, y=None):  # noqa: ANN001, ANN201
        X_arr = check_array(X, ensure_2d=True, dtype=np.float64)
        self._set_n_classes(y)

        n = int(X_arr.shape[0])
        if n == 0:
            raise ValueError("Training set cannot be empty")

        r = float(self.coreset_ratio)
        if not (0.0 < r <= 1.0):
            raise ValueError(f"coreset_ratio must be in (0,1], got {self.coreset_ratio}")

        m = max(1, int(np.ceil(r * n)))
        rng = np.random.RandomState(0 if self.random_state is None else int(self.random_state))
        idx = rng.choice(n, size=m, replace=False).astype(np.int64, copy=False)
        bank = np.asarray(X_arr[idx], dtype=np.float64)

        nn = NearestNeighbors(n_neighbors=2 if m > 1 else 1, metric=self.metric, p=self.p)
        nn.fit(bank)

        # Training scores: distance to nearest bank item, using 2-NN to avoid self==0
        dist, _ = nn.kneighbors(X_arr, n_neighbors=2 if m > 1 else 1, return_distance=True)
        dist = np.asarray(dist, dtype=np.float64)
        if m > 1:
            # If a point is in the bank, 1-NN distance can be 0; use the 2nd NN.
            d0 = dist[:, 0]
            d1 = dist[:, 1]
            score = np.where(d0 <= 0.0, d1, d0)
        else:
            score = dist[:, 0]

        self.memory_bank_ = bank
        self.nn_ = nn
        self.decision_scores_ = np.asarray(score, dtype=np.float64).reshape(-1)
        self._process_decision_scores()
        return self

    def decision_function(self, X):  # noqa: ANN001, ANN201
        require_fitted(self, ["nn_"])
        X_arr = check_array(X, ensure_2d=True, dtype=np.float64)
        nn: NearestNeighbors = self.nn_  # type: ignore[assignment]
        dist, _ = nn.kneighbors(X_arr, n_neighbors=1, return_distance=True)
        return np.asarray(dist[:, 0], dtype=np.float64).reshape(-1)


@register_model(
    "vision_patchcore_lite",
    tags=("vision", "classical", "neighbors", "memory_bank", "patchcore"),
    metadata={
        "description": "PatchCore-lite: embedding extractor + coreset memory bank + NN distance (image-level)",
    },
)
class VisionPatchCoreLite(BaseVisionDetector):
    def __init__(
        self,
        *,
        feature_extractor=None,
        contamination: float = 0.1,
        coreset_ratio: float = 0.2,
        metric: str = "minkowski",
        p: int = 2,
        random_state: Optional[int] = 0,
    ) -> None:
        self._detector_kwargs = {
            "contamination": float(contamination),
            "coreset_ratio": float(coreset_ratio),
            "metric": str(metric),
            "p": int(p),
            "random_state": random_state,
        }
        super().__init__(contamination=contamination, feature_extractor=feature_extractor)

    def _build_detector(self):
        return CorePatchCoreLite(**self._detector_kwargs)
