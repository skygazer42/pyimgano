# -*- coding: utf-8 -*-
"""RRCF (Robust Random Cut Forest) - lightweight variant.

This module implements a simple random cut forest with an isolation-depth style
scoring. While full RRCF uses collusive displacement, the random cut tree
construction itself is a useful baseline for fast, dependency-stable anomaly
scoring.

Score: average( 1 / (path_length + 1) ) across trees.
Higher score => isolated earlier => more anomalous.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.utils.validation import check_array

from .base_detector import BaseDetector
from .baseml import BaseVisionDetector
from .registry import register_model
from ..utils.fitted import require_fitted
from ..utils.random_state import check_random_state

logger = logging.getLogger(__name__)


@dataclass
class _RCTNode:
    split_dim: int | None = None
    split_val: float | None = None
    left: "_RCTNode | None" = None
    right: "_RCTNode | None" = None


def _build_random_cut_tree(
    X: np.ndarray,
    idxs: np.ndarray,
    *,
    max_depth: int,
    rng: np.random.RandomState,
    depth: int = 0,
    max_tries: int = 8,
) -> _RCTNode:
    if idxs.size <= 1 or depth >= max_depth:
        return _RCTNode()

    pts = X[idxs]
    lo = np.min(pts, axis=0)
    hi = np.max(pts, axis=0)
    ranges = hi - lo
    valid_dims = np.flatnonzero(ranges > 0.0)
    if valid_dims.size == 0:
        return _RCTNode()

    for _ in range(max_tries):
        dim = int(rng.choice(valid_dims))
        cut = float(rng.uniform(lo[dim], hi[dim]))
        mask_left = pts[:, dim] < cut
        if not mask_left.any() or mask_left.all():
            continue
        left_idxs = idxs[mask_left]
        right_idxs = idxs[~mask_left]
        return _RCTNode(
            split_dim=dim,
            split_val=cut,
            left=_build_random_cut_tree(
                X, left_idxs, max_depth=max_depth, rng=rng, depth=depth + 1
            ),
            right=_build_random_cut_tree(
                X, right_idxs, max_depth=max_depth, rng=rng, depth=depth + 1
            ),
        )

    # Fall back to leaf if we failed to split (e.g., repeated degenerate cuts).
    return _RCTNode()


def _path_length(root: _RCTNode, x: np.ndarray) -> int:
    node = root
    depth = 0
    while node.split_dim is not None and node.left is not None and node.right is not None:
        if x[node.split_dim] < float(node.split_val):
            node = node.left
        else:
            node = node.right
        depth += 1
    return depth


@register_model(
    "core_rrcf",
    tags=("classical", "forest", "random-cut"),
    metadata={"description": "Random cut forest baseline (RRCF-style tree construction)"},
)
class CoreRRCF(BaseDetector):
    def __init__(
        self,
        *,
        contamination: float = 0.1,
        n_trees: int = 50,
        max_samples: int = 256,
        max_depth: int | None = None,
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        super().__init__(contamination=float(contamination))
        self.n_trees = int(n_trees)
        self.max_samples = int(max_samples)
        self.max_depth = None if max_depth is None else int(max_depth)
        self.random_state = random_state

    def fit(self, X, y=None):  # noqa: ANN001, ANN201
        X_arr = check_array(X, ensure_2d=True, dtype=np.float64)
        self._set_n_classes(y)

        n = int(X_arr.shape[0])
        if n == 0:
            raise ValueError("X must be non-empty")
        if self.n_trees <= 0:
            raise ValueError("n_trees must be > 0")

        sample_size = min(n, max(2, int(self.max_samples)))
        max_depth = self.max_depth
        if max_depth is None:
            # A reasonable default for random partition trees.
            max_depth = max(1, int(np.ceil(np.log2(sample_size))) + 1)

        rng = check_random_state(self.random_state)
        forest: list[_RCTNode] = []
        for _ in range(int(self.n_trees)):
            if sample_size < n:
                idxs = rng.choice(n, size=sample_size, replace=False)
            else:
                idxs = np.arange(n, dtype=np.int64)
            root = _build_random_cut_tree(X_arr, np.asarray(idxs, dtype=np.int64), max_depth=max_depth, rng=rng)
            forest.append(root)

        self._forest = forest
        self._X_train = X_arr

        self.decision_scores_ = np.asarray(self.decision_function(X_arr), dtype=np.float64)
        self._process_decision_scores()
        return self

    def decision_function(self, X):  # noqa: ANN001, ANN201
        require_fitted(self, ["_forest"])
        forest: list[_RCTNode] = self._forest  # type: ignore[assignment]

        X_arr = check_array(X, ensure_2d=True, dtype=np.float64)
        if X_arr.shape[0] == 0:
            return np.zeros((0,), dtype=np.float64)

        scores = np.zeros((X_arr.shape[0],), dtype=np.float64)
        for t in forest:
            depths = np.asarray([_path_length(t, X_arr[i]) for i in range(X_arr.shape[0])], dtype=np.float64)
            scores += 1.0 / (depths + 1.0)
        scores /= float(len(forest))
        return scores


@register_model(
    "vision_rrcf",
    tags=("vision", "classical", "forest", "random-cut"),
    metadata={"description": "Vision random cut forest baseline (RRCF-style)"},
)
class VisionRRCF(BaseVisionDetector):
    def __init__(
        self,
        *,
        feature_extractor=None,
        contamination: float = 0.1,
        n_trees: int = 50,
        max_samples: int = 256,
        max_depth: int | None = None,
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        self._detector_kwargs = {
            "contamination": float(contamination),
            "n_trees": int(n_trees),
            "max_samples": int(max_samples),
            "max_depth": (None if max_depth is None else int(max_depth)),
            "random_state": random_state,
        }
        logger.debug("Initializing VisionRRCF with kwargs=%s", self._detector_kwargs)
        super().__init__(contamination=contamination, feature_extractor=feature_extractor)

    def _build_detector(self):
        return CoreRRCF(**self._detector_kwargs)

