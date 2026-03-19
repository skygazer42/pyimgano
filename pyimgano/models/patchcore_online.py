# -*- coding: utf-8 -*-
"""PatchCore-online (incremental memory bank update; study-only v1).

This is a lightweight, industrially useful variant inspired by PatchCore-lite:

  embeddings -> memory bank -> nearest-neighbor distance score

Unlike full PatchCore, this implementation is **image-level only** (no pixel map).

The "online" part:
- supports `partial_fit(X_new)` to append new normal samples and rebuild the NN index
- optionally caps bank size to keep latency bounded
"""

from __future__ import annotations

from typing import Optional, cast

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_array

from pyimgano.utils.fitted import require_fitted

from ._legacy_x import MISSING, resolve_legacy_x_keyword
from .base_detector import BaseDetector
from .baseml import BaseVisionDetector
from .registry import register_model


@register_model(
    "core_patchcore_online",
    tags=("classical", "core", "features", "neighbors", "memory_bank", "patchcore", "online"),
    metadata={
        "description": "PatchCore-online: incremental memory bank + nearest-neighbor distance (image-level)",
        "paper": "Towards Total Recall in Industrial Anomaly Detection",
        "year": 2022,
        "type": "neighbors",
    },
)
class CorePatchCoreOnline(BaseDetector):
    def __init__(
        self,
        *,
        contamination: float = 0.1,
        metric: str = "minkowski",
        p: int = 2,
        max_bank_size: int | None = 10_000,
        random_state: Optional[int] = 0,
    ) -> None:
        super().__init__(contamination=float(contamination))
        self.metric = str(metric)
        self.p = int(p)
        self.max_bank_size = None if max_bank_size is None else int(max_bank_size)
        self.random_state = random_state

        self.memory_bank_: np.ndarray | None = None
        self.nn_: NearestNeighbors | None = None

    def _rebuild_index(self) -> None:
        if self.memory_bank_ is None:
            raise RuntimeError("Internal error: missing memory_bank_")
        bank = np.asarray(self.memory_bank_, dtype=np.float64)
        nn = NearestNeighbors(
            n_neighbors=1 if bank.shape[0] <= 1 else 2, metric=self.metric, p=self.p
        )
        nn.fit(bank)
        self.nn_ = nn

    def _maybe_cap_bank(self) -> None:
        if self.memory_bank_ is None:
            return
        if self.max_bank_size is None:
            return
        max_n = int(self.max_bank_size)
        if max_n <= 0:
            raise ValueError("max_bank_size must be positive or None")
        n = int(self.memory_bank_.shape[0])
        if n <= max_n:
            return

        rng = np.random.default_rng(0 if self.random_state is None else int(self.random_state))
        idx = rng.choice(n, size=max_n, replace=False).astype(np.int64, copy=False)
        self.memory_bank_ = np.asarray(self.memory_bank_[idx], dtype=np.float64)

    def fit(self, x: object = MISSING, y=None, **kwargs: object):  # noqa: ANN001, ANN201
        x_value = resolve_legacy_x_keyword(x, kwargs, method_name="fit")
        x_arr = check_array(cast(object, x_value), ensure_2d=True, dtype=np.float64)
        self._set_n_classes(y)

        if int(x_arr.shape[0]) == 0:
            raise ValueError("Training set cannot be empty")

        self.memory_bank_ = np.asarray(x_arr, dtype=np.float64)
        self._maybe_cap_bank()
        self._rebuild_index()

        # Training scores: use 2-NN when possible to avoid self==0.
        nn = self.nn_
        assert nn is not None
        n = int(self.memory_bank_.shape[0])
        dist, _ = nn.kneighbors(x_arr, n_neighbors=2 if n > 1 else 1, return_distance=True)
        dist = np.asarray(dist, dtype=np.float64)
        if n > 1 and dist.shape[1] >= 2:
            d0 = dist[:, 0]
            d1 = dist[:, 1]
            score = np.where(d0 <= 0.0, d1, d0)
        else:
            score = dist[:, 0]

        self.decision_scores_ = np.asarray(score, dtype=np.float64).reshape(-1)
        self._process_decision_scores()
        return self

    def partial_fit(self, x: object = MISSING, y=None, **kwargs: object):  # noqa: ANN001, ANN201
        x_value = resolve_legacy_x_keyword(x, kwargs, method_name="partial_fit")
        x_arr = check_array(cast(object, x_value), ensure_2d=True, dtype=np.float64)
        if int(x_arr.shape[0]) == 0:
            return self

        if self.memory_bank_ is None:
            return self.fit(x_arr, y=y)

        self.memory_bank_ = np.concatenate(
            [np.asarray(self.memory_bank_, dtype=np.float64), x_arr], axis=0
        )
        self._maybe_cap_bank()
        self._rebuild_index()
        return self

    def decision_function(self, x: object = MISSING, **kwargs: object):  # noqa: ANN001, ANN201
        require_fitted(self, ["nn_"])
        x_value = resolve_legacy_x_keyword(x, kwargs, method_name="decision_function")
        x_arr = check_array(cast(object, x_value), ensure_2d=True, dtype=np.float64)
        nn: NearestNeighbors = self.nn_  # type: ignore[assignment]
        dist, _ = nn.kneighbors(x_arr, n_neighbors=1, return_distance=True)
        return np.asarray(dist[:, 0], dtype=np.float64).reshape(-1)


@register_model(
    "vision_patchcore_online",
    tags=("vision", "classical", "neighbors", "memory_bank", "patchcore", "online"),
    metadata={
        "description": "PatchCore-online: feature extractor + incremental memory bank + NN distance (image-level)",
        "paper": "Towards Total Recall in Industrial Anomaly Detection",
        "year": 2022,
    },
)
class VisionPatchCoreOnline(BaseVisionDetector):
    def __init__(
        self,
        *,
        feature_extractor=None,
        contamination: float = 0.1,
        metric: str = "minkowski",
        p: int = 2,
        max_bank_size: int | None = 10_000,
        random_state: Optional[int] = 0,
    ) -> None:
        self._detector_kwargs = {
            "contamination": float(contamination),
            "metric": str(metric),
            "p": int(p),
            "max_bank_size": (None if max_bank_size is None else int(max_bank_size)),
            "random_state": random_state,
        }
        super().__init__(contamination=contamination, feature_extractor=feature_extractor)

    def _build_detector(self):
        return CorePatchCoreOnline(**self._detector_kwargs)

    def partial_fit(self, x: object = MISSING, y=None, **kwargs: object):  # noqa: ANN001, ANN201
        x_value = resolve_legacy_x_keyword(x, kwargs, method_name="partial_fit")
        feats = self.feature_extractor.extract(x_value)
        pf = getattr(self.detector, "partial_fit", None)
        if not callable(pf):
            raise RuntimeError("Internal error: core detector does not support partial_fit")
        pf(feats, y=y)
        return self


__all__ = ["CorePatchCoreOnline", "VisionPatchCoreOnline"]
