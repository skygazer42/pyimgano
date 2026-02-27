# -*- coding: utf-8 -*-
"""Random-trees density baseline (ExtraTrees-style embedding).

We fit a RandomTreesEmbedding (unsupervised tree partitioning) and score points
by how rare their leaf assignments are across the forest.

Score definition
---------------
For each tree:
  contrib = -log( leaf_count / n_train )
Score = mean(contrib over trees)

Higher score => more anomalous.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.utils.validation import check_array

from pyimgano.utils.fitted import require_fitted

from .base_detector import BaseDetector
from .baseml import BaseVisionDetector
from .registry import register_model


@register_model(
    "core_extra_trees_density",
    tags=("classical", "core", "features", "trees", "density"),
    metadata={
        "description": "RandomTreesEmbedding leaf-rarity density baseline (native)",
        "type": "density",
    },
)
class CoreExtraTreesDensity(BaseDetector):
    def __init__(
        self,
        *,
        contamination: float = 0.1,
        n_estimators: int = 50,
        max_depth: int = 5,
        min_samples_leaf: int = 1,
        random_state: int | None = 0,
        eps: float = 1e-12,
    ) -> None:
        super().__init__(contamination=float(contamination))
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)
        self.random_state = random_state
        self.eps = float(eps)

    def fit(self, X, y=None):  # noqa: ANN001, ANN201
        X_arr = check_array(X, ensure_2d=True, dtype=np.float64)
        self._set_n_classes(y)

        emb = RandomTreesEmbedding(
            n_estimators=int(self.n_estimators),
            max_depth=int(self.max_depth),
            min_samples_leaf=int(self.min_samples_leaf),
            random_state=self.random_state,
        )
        emb.fit(X_arr)

        leaf = emb.apply(X_arr)  # (n, n_estimators)
        leaf = np.asarray(leaf, dtype=np.int64)
        n = int(leaf.shape[0])
        m = int(leaf.shape[1])
        if n == 0 or m == 0:
            self.embedding_ = emb
            self.leaf_counts_ = []
            self.decision_scores_ = np.zeros((n,), dtype=np.float64)
            self._process_decision_scores()
            return self

        counts_per_tree: list[dict[int, int]] = []
        for t in range(m):
            col = leaf[:, t]
            uniq, cnt = np.unique(col, return_counts=True)
            counts_per_tree.append({int(u): int(c) for u, c in zip(uniq, cnt, strict=False)})

        self.embedding_ = emb
        self.leaf_counts_ = counts_per_tree

        scores = self._score_from_leaf(leaf)
        self.decision_scores_ = scores
        self._process_decision_scores()
        return self

    def _score_from_leaf(self, leaf: np.ndarray) -> np.ndarray:
        leaf = np.asarray(leaf, dtype=np.int64)
        n = int(leaf.shape[0])
        m = int(leaf.shape[1]) if leaf.ndim == 2 else 0
        if n == 0 or m == 0:
            return np.zeros((n,), dtype=np.float64)

        counts_per_tree: list[dict[int, int]] = self.leaf_counts_  # type: ignore[attr-defined]
        inv_n = 1.0 / max(float(n), float(self.eps))

        out = np.zeros((n,), dtype=np.float64)
        for t in range(m):
            cdict = counts_per_tree[t]
            col = leaf[:, t]
            # Unseen leaf => count=1 (rare) for out-of-sample scoring.
            cnt = np.asarray([cdict.get(int(v), 1) for v in col], dtype=np.float64)
            freq = np.maximum(cnt * inv_n, float(self.eps))
            out += -np.log(freq)
        out = out / float(m)
        return np.asarray(out, dtype=np.float64).reshape(-1)

    def decision_function(self, X):  # noqa: ANN001, ANN201
        require_fitted(self, ["embedding_", "leaf_counts_"])
        X_arr = check_array(X, ensure_2d=True, dtype=np.float64)
        emb: RandomTreesEmbedding = self.embedding_  # type: ignore[assignment]
        leaf = np.asarray(emb.apply(X_arr), dtype=np.int64)
        return self._score_from_leaf(leaf)


@register_model(
    "vision_extra_trees_density",
    tags=("vision", "classical", "trees", "density"),
    metadata={"description": "Vision wrapper for random-trees leaf-rarity density baseline"},
)
class VisionExtraTreesDensity(BaseVisionDetector):
    def __init__(
        self,
        *,
        feature_extractor=None,
        contamination: float = 0.1,
        n_estimators: int = 50,
        max_depth: int = 5,
        min_samples_leaf: int = 1,
        random_state: int | None = 0,
        eps: float = 1e-12,
    ) -> None:
        self._detector_kwargs = {
            "contamination": float(contamination),
            "n_estimators": int(n_estimators),
            "max_depth": int(max_depth),
            "min_samples_leaf": int(min_samples_leaf),
            "random_state": random_state,
            "eps": float(eps),
        }
        super().__init__(contamination=contamination, feature_extractor=feature_extractor)

    def _build_detector(self):
        return CoreExtraTreesDensity(**self._detector_kwargs)
