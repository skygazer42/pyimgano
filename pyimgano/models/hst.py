# -*- coding: utf-8 -*-
"""Half-Space Trees (HST) - lightweight batch implementation.

Half-Space Trees are an online anomaly detector that partitions feature space
with random half-space cuts and uses leaf mass as an outlier signal.

This implementation builds random trees over the training feature ranges and
scores samples by inverse leaf mass:
    score = mean_trees( 1 / (leaf_count + 1) )
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
class _HSTNode:
    split_dim: int | None = None
    split_val: float | None = None
    left: "_HSTNode | None" = None
    right: "_HSTNode | None" = None
    count: int = 0


def _build_hst(
    lo: np.ndarray,
    hi: np.ndarray,
    *,
    depth: int,
    max_depth: int,
    rng: np.random.RandomState,
) -> _HSTNode:
    if depth >= max_depth:
        return _HSTNode()

    d = int(lo.shape[0])
    dim = int(rng.randint(0, d))
    a = float(lo[dim])
    b = float(hi[dim])
    if b <= a:
        cut = a
    else:
        cut = float(rng.uniform(a, b))

    lo_l = lo.copy()
    hi_l = hi.copy()
    hi_l[dim] = cut

    lo_r = lo.copy()
    hi_r = hi.copy()
    lo_r[dim] = cut

    return _HSTNode(
        split_dim=dim,
        split_val=cut,
        left=_build_hst(lo_l, hi_l, depth=depth + 1, max_depth=max_depth, rng=rng),
        right=_build_hst(lo_r, hi_r, depth=depth + 1, max_depth=max_depth, rng=rng),
    )


def _leaf(node: _HSTNode, x: np.ndarray, *, max_depth: int) -> _HSTNode:
    cur = node
    depth = 0
    while (
        depth < max_depth
        and cur.split_dim is not None
        and cur.left is not None
        and cur.right is not None
    ):
        if x[cur.split_dim] < float(cur.split_val):
            cur = cur.left
        else:
            cur = cur.right
        depth += 1
    return cur


@register_model(
    "core_hst",
    tags=("classical", "tree", "online"),
    metadata={"description": "Half-Space Trees (leaf-mass scoring; native)"},
)
class CoreHST(BaseDetector):
    def __init__(
        self,
        *,
        contamination: float = 0.1,
        n_trees: int = 25,
        max_depth: int = 10,
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        super().__init__(contamination=float(contamination))
        self.n_trees = int(n_trees)
        self.max_depth = int(max_depth)
        self.random_state = random_state

    def fit(self, X, y=None):  # noqa: ANN001, ANN201
        X_arr = check_array(X, ensure_2d=True, dtype=np.float64)
        self._set_n_classes(y)

        n, d = X_arr.shape
        if n == 0:
            raise ValueError("X must be non-empty")
        if self.n_trees <= 0:
            raise ValueError("n_trees must be > 0")
        if self.max_depth <= 0:
            raise ValueError("max_depth must be > 0")

        lo = np.min(X_arr, axis=0)
        hi = np.max(X_arr, axis=0)
        # Avoid degenerate ranges that would prevent meaningful cuts.
        same = hi <= lo
        if np.any(same):
            hi = hi.copy()
            hi[same] = lo[same] + 1.0

        rng = check_random_state(self.random_state)
        forest: list[_HSTNode] = []
        for _ in range(int(self.n_trees)):
            root = _build_hst(lo, hi, depth=0, max_depth=int(self.max_depth), rng=rng)
            forest.append(root)

        # Count leaf masses.
        for root in forest:
            for i in range(n):
                leaf = _leaf(root, X_arr[i], max_depth=int(self.max_depth))
                leaf.count += 1

        self._forest = forest
        self._lo = lo
        self._hi = hi

        self.decision_scores_ = np.asarray(self.decision_function(X_arr), dtype=np.float64)
        self._process_decision_scores()
        return self

    def decision_function(self, X):  # noqa: ANN001, ANN201
        require_fitted(self, ["_forest"])
        forest: list[_HSTNode] = self._forest  # type: ignore[assignment]

        X_arr = check_array(X, ensure_2d=True, dtype=np.float64)
        if X_arr.shape[0] == 0:
            return np.zeros((0,), dtype=np.float64)

        scores = np.zeros((X_arr.shape[0],), dtype=np.float64)
        for root in forest:
            counts = np.asarray(
                [_leaf(root, X_arr[i], max_depth=int(self.max_depth)).count for i in range(X_arr.shape[0])],
                dtype=np.float64,
            )
            scores += 1.0 / (counts + 1.0)
        scores /= float(len(forest))
        return scores


@register_model(
    "vision_hst",
    tags=("vision", "classical", "tree", "online"),
    metadata={"description": "Vision Half-Space Trees (leaf-mass scoring)"},
)
class VisionHST(BaseVisionDetector):
    def __init__(
        self,
        *,
        feature_extractor=None,
        contamination: float = 0.1,
        n_trees: int = 25,
        max_depth: int = 10,
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        self._detector_kwargs = {
            "contamination": float(contamination),
            "n_trees": int(n_trees),
            "max_depth": int(max_depth),
            "random_state": random_state,
        }
        logger.debug("Initializing VisionHST with kwargs=%s", self._detector_kwargs)
        super().__init__(contamination=contamination, feature_extractor=feature_extractor)

    def _build_detector(self):
        return CoreHST(**self._detector_kwargs)

