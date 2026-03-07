# -*- coding: utf-8 -*-
"""MST-based outlier detection.

This is a lightweight, industrial-friendly baseline:
- Build a Minimum Spanning Tree (MST) on training embeddings
- Score each training point by its maximum incident MST edge length
- Score new points by distance to nearest training point (optionally combined)

Higher score => more anomalous.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_array

from pyimgano.utils.fitted import require_fitted

from .base_detector import BaseDetector
from .baseml import BaseVisionDetector
from .registry import register_model


def _mst_node_scores_from_distance_matrix(D: np.ndarray) -> np.ndarray:
    """Compute per-node MST score = max incident edge length (undirected)."""

    # Local import: keep import surface lighter.
    from scipy.sparse.csgraph import minimum_spanning_tree

    n = int(D.shape[0])
    if n <= 1:
        return np.zeros((n,), dtype=np.float64)

    mst = minimum_spanning_tree(np.asarray(D, dtype=np.float64))
    coo = mst.tocoo()

    scores = np.zeros((n,), dtype=np.float64)
    # NOTE: zip(strict=...) is Python 3.10+. We support Python 3.9 (see requires-python),
    # so keep the default (non-strict) zip behavior here.
    for i, j, w in zip(coo.row, coo.col, coo.data):
        wi = float(w)
        if wi > scores[int(i)]:
            scores[int(i)] = wi
        if wi > scores[int(j)]:
            scores[int(j)] = wi
    return scores


@register_model(
    "core_mst_outlier",
    tags=("classical", "core", "features", "graph", "mst"),
    metadata={
        "description": "MST-based outlier baseline (max incident MST edge length)",
        "type": "graph",
    },
)
class CoreMSTOutlier(BaseDetector):
    def __init__(
        self,
        *,
        contamination: float = 0.1,
        metric: str = "euclidean",
        score_mode: Literal["nn", "max"] = "max",
    ) -> None:
        super().__init__(contamination=float(contamination))
        self.metric = str(metric)
        self.score_mode = str(score_mode)

    def fit(self, X, y=None):  # noqa: ANN001, ANN201
        X_arr = check_array(X, ensure_2d=True, dtype=np.float64)
        self._set_n_classes(y)

        n = int(X_arr.shape[0])
        if n <= 1:
            self._X_train = X_arr
            self._nn = NearestNeighbors(n_neighbors=1, metric=self.metric).fit(X_arr)
            self._node_scores = np.zeros((n,), dtype=np.float64)
            self.decision_scores_ = np.zeros((n,), dtype=np.float64)
            self._process_decision_scores()
            return self

        # Compute pairwise distances on train set.
        from sklearn.metrics import pairwise_distances

        D = pairwise_distances(X_arr, metric=self.metric)
        # Force a stable diagonal.
        np.fill_diagonal(D, 0.0)

        node_scores = _mst_node_scores_from_distance_matrix(D)

        self._X_train = X_arr
        self._node_scores = np.asarray(node_scores, dtype=np.float64).reshape(-1)
        self._nn = NearestNeighbors(n_neighbors=1, metric=self.metric).fit(X_arr)

        # Training score definition: node MST score.
        self.decision_scores_ = np.asarray(self._node_scores, dtype=np.float64).reshape(-1)
        self._process_decision_scores()
        return self

    def decision_function(self, X):  # noqa: ANN001, ANN201
        require_fitted(self, ["_X_train", "_nn", "_node_scores"])
        X_arr = check_array(X, ensure_2d=True, dtype=np.float64)

        nn: NearestNeighbors = self._nn  # type: ignore[assignment]
        dist, ind = nn.kneighbors(X_arr, n_neighbors=1, return_distance=True)
        dist = np.asarray(dist, dtype=np.float64).reshape(-1)
        ind = np.asarray(ind, dtype=np.int64).reshape(-1)

        base = dist
        if str(self.score_mode).lower().strip() == "max":
            node_scores = np.asarray(self._node_scores, dtype=np.float64).reshape(-1)  # type: ignore[arg-type]
            base = np.maximum(base, node_scores[ind])

        return np.asarray(base, dtype=np.float64).reshape(-1)


@register_model(
    "vision_mst_outlier",
    tags=("vision", "classical", "graph", "mst"),
    metadata={"description": "Vision wrapper for MST-based outlier detector"},
)
class VisionMSTOutlier(BaseVisionDetector):
    def __init__(
        self,
        *,
        feature_extractor=None,
        contamination: float = 0.1,
        metric: str = "euclidean",
        score_mode: Literal["nn", "max"] = "max",
    ) -> None:
        self._detector_kwargs = {
            "contamination": float(contamination),
            "metric": str(metric),
            "score_mode": str(score_mode),
        }
        super().__init__(contamination=contamination, feature_extractor=feature_extractor)

    def _build_detector(self):
        return CoreMSTOutlier(**self._detector_kwargs)
