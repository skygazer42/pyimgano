# -*- coding: utf-8 -*-
"""Robust Random Cut Forest with collusive-displacement scoring.

Random-cut dimensions are sampled in proportion to their observed range and
training points are scored by collusive displacement.  New samples use a
deterministic fitted-tree insertion-path extension because the paper's stream
algorithm defines scores while inserting/removing points rather than a sklearn
``decision_function``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from sklearn.utils.validation import check_array

from ..utils.fitted import require_fitted
from ..utils.random_state import check_random_state
from .base_detector import BaseDetector
from .baseml import BaseVisionDetector
from .registry import register_model

logger = logging.getLogger(__name__)


@dataclass
class _RCTNode:
    split_dim: int | None = None
    split_val: float | None = None
    left: "_RCTNode | None" = None
    right: "_RCTNode | None" = None
    parent: "_RCTNode | None" = field(default=None, repr=False)
    size: int = 0
    bbox_min: np.ndarray | None = field(default=None, repr=False)
    bbox_max: np.ndarray | None = field(default=None, repr=False)
    sample_indices: tuple[int, ...] = ()


@dataclass
class _RandomCutTree:
    root: _RCTNode
    leaves: dict[int, _RCTNode]


def _build_random_cut_tree(
    x: np.ndarray,
    idxs: np.ndarray,
    *,
    max_depth: int | None,
    rng: np.random.Generator,
    depth: int = 0,
) -> _RCTNode:
    pts = x[idxs]
    lo = np.min(pts, axis=0)
    hi = np.max(pts, axis=0)
    node = _RCTNode(
        size=int(idxs.size),
        bbox_min=np.asarray(lo, dtype=np.float64),
        bbox_max=np.asarray(hi, dtype=np.float64),
    )
    if idxs.size <= 1 or (max_depth is not None and depth >= max_depth):
        node.sample_indices = tuple(int(index) for index in idxs)
        return node

    ranges = hi - lo
    valid_dims = np.flatnonzero(ranges > 0.0)
    if valid_dims.size == 0:
        node.sample_indices = tuple(int(index) for index in idxs)
        return node

    probabilities = ranges[valid_dims] / float(np.sum(ranges[valid_dims]))
    dim = int(rng.choice(valid_dims, p=probabilities))
    cut = float(rng.uniform(lo[dim], hi[dim]))
    mask_left = pts[:, dim] <= cut
    left_idxs = idxs[mask_left]
    right_idxs = idxs[~mask_left]
    # A continuous cut over a positive range should split both sides; retain a
    # deterministic midpoint fallback for adversarial RNG implementations.
    if left_idxs.size == 0 or right_idxs.size == 0:
        cut = float((lo[dim] + hi[dim]) / 2.0)
        mask_left = pts[:, dim] <= cut
        left_idxs = idxs[mask_left]
        right_idxs = idxs[~mask_left]
    if left_idxs.size == 0 or right_idxs.size == 0:
        node.sample_indices = tuple(int(index) for index in idxs)
        return node

    node.split_dim = dim
    node.split_val = cut
    node.left = _build_random_cut_tree(x, left_idxs, max_depth=max_depth, rng=rng, depth=depth + 1)
    node.right = _build_random_cut_tree(
        x, right_idxs, max_depth=max_depth, rng=rng, depth=depth + 1
    )
    node.left.parent = node
    node.right.parent = node
    return node


def _leaf_map(root: _RCTNode) -> dict[int, _RCTNode]:
    leaves: dict[int, _RCTNode] = {}
    stack = [root]
    while stack:
        node = stack.pop()
        if node.left is None or node.right is None:
            for sample_index in node.sample_indices:
                leaves[int(sample_index)] = node
            continue
        stack.extend((node.left, node.right))
    return leaves


def _collusive_displacement(leaf: _RCTNode) -> float:
    """Return max sibling displacement divided by the current subtree mass."""

    node = leaf
    score = 0.0
    while node.parent is not None:
        parent = node.parent
        sibling = parent.right if parent.left is node else parent.left
        if sibling is None:
            break
        score = max(score, float(sibling.size) / float(max(node.size, 1)))
        node = parent
    return score


def _point_outside(node: _RCTNode, point: np.ndarray) -> bool:
    if node.bbox_min is None or node.bbox_max is None:
        return False
    return bool(np.any(point < node.bbox_min) or np.any(point > node.bbox_max))


def _insertion_path_codisp(root: _RCTNode, point: np.ndarray) -> float:
    """Deterministic novelty score induced by inserting a new leaf."""

    if _point_outside(root, point):
        return float(root.size)

    node = root
    sibling_sizes: list[int] = []
    while node.split_dim is not None and node.left is not None and node.right is not None:
        if point[node.split_dim] <= float(node.split_val):
            child, sibling = node.left, node.right
        else:
            child, sibling = node.right, node.left
        if _point_outside(child, point):
            return max(float(child.size), float(max(sibling_sizes, default=0)))
        sibling_sizes.append(int(sibling.size))
        node = child

    # The new singleton first displaces the terminal leaf.  Moving upward, its
    # containing subtree grows by each encountered sibling mass.
    score = float(node.size)
    current_mass = float(node.size + 1)
    for sibling_size in reversed(sibling_sizes):
        score = max(score, float(sibling_size) / current_mass)
        current_mass += float(sibling_size)
    return score


@register_model(
    "core_rrcf",
    tags=("classical", "core", "features", "forest", "random-cut"),
    metadata={
        "description": "Range-proportional random-cut forest with collusive displacement",
        "paper": "Robust Random Cut Forest Based Anomaly Detection on Streams",
        "paper_url": "https://proceedings.mlr.press/v48/guha16.html",
        "year": 2016,
        "paper_fidelity": "paper-adaptation",
        "implementation_status": "range-proportional-rct-codisp-with-fitted-novelty-extension",
        "known_deviation": "decision_function uses a deterministic insertion-path novelty extension for samples outside the fitted stream.",
    },
)
class CoreRRCF(BaseDetector):
    def __init__(
        self,
        *,
        contamination: float = 0.1,
        n_trees: int = 50,
        max_samples: int = 256,
        max_depth: int | None = None,
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        super().__init__(contamination=float(contamination))
        self.n_trees = int(n_trees)
        self.max_samples = int(max_samples)
        self.max_depth = None if max_depth is None else int(max_depth)
        self.random_state = random_state

    def fit(self, x, y=None):  # noqa: ANN001, ANN201
        x_arr = check_array(x, ensure_2d=True, dtype=np.float64)
        self._set_n_classes(y)

        n = int(x_arr.shape[0])
        if n == 0:
            raise ValueError("x must be non-empty")
        if self.n_trees <= 0:
            raise ValueError("n_trees must be > 0")

        sample_size = min(n, max(2, int(self.max_samples)))
        max_depth = self.max_depth
        if max_depth is not None and max_depth < 1:
            raise ValueError("max_depth must be >= 1 when provided")

        rng = check_random_state(self.random_state)
        forest: list[_RandomCutTree] = []
        train_scores = np.zeros((n,), dtype=np.float64)
        for _ in range(int(self.n_trees)):
            if sample_size < n:
                idxs = rng.choice(n, size=sample_size, replace=False)
            else:
                idxs = np.arange(n, dtype=np.int64)
            root = _build_random_cut_tree(
                x_arr, np.asarray(idxs, dtype=np.int64), max_depth=max_depth, rng=rng
            )
            leaves = _leaf_map(root)
            forest.append(_RandomCutTree(root=root, leaves=leaves))
            for sample_index in range(n):
                leaf = leaves.get(sample_index)
                if leaf is not None:
                    train_scores[sample_index] += _collusive_displacement(leaf)
                else:
                    train_scores[sample_index] += _insertion_path_codisp(root, x_arr[sample_index])

        self._forest = forest
        self._X_train = x_arr

        self.decision_scores_ = train_scores / float(len(forest))
        self._process_decision_scores()
        return self

    def decision_function(self, x):  # noqa: ANN001, ANN201
        require_fitted(self, ["_forest"])
        forest: list[_RandomCutTree] = self._forest  # type: ignore[assignment]

        x_arr = check_array(x, ensure_2d=True, dtype=np.float64)
        if x_arr.shape[0] == 0:
            return np.zeros((0,), dtype=np.float64)

        scores = np.zeros((x_arr.shape[0],), dtype=np.float64)
        for tree in forest:
            scores += np.asarray(
                [_insertion_path_codisp(tree.root, point) for point in x_arr],
                dtype=np.float64,
            )
        scores /= float(len(forest))
        return scores


@register_model(
    "vision_rrcf",
    tags=("vision", "classical", "forest", "random-cut"),
    metadata={
        "description": "Vision wrapper for range-proportional RRCF with codisp scoring",
        "paper": "Robust Random Cut Forest Based Anomaly Detection on Streams",
        "paper_url": "https://proceedings.mlr.press/v48/guha16.html",
        "year": 2016,
        "paper_fidelity": "paper-adaptation",
        "implementation_status": "vision-wrapper-over-rct-codisp-novelty-extension",
        "known_deviation": "Out-of-sample scoring is a fitted insertion-path extension of the streaming algorithm.",
    },
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
        random_state: int | np.random.Generator | None = None,
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
