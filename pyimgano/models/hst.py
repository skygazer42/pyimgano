# -*- coding: utf-8 -*-
"""Half-Space Trees for batch fitting and online window updates.

The tree workspaces, midpoint cuts, mass profiles, and terminal-node score
follow Algorithm 1--3 of Tan et al. (IJCAI 2011). The paper's small score
means anomalous; this module exposes its reciprocal so that the package-wide
contract remains ``larger score == more anomalous``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.utils.validation import check_array

from ..utils.fitted import require_fitted
from ..utils.random_state import check_random_state
from ._legacy_x import MISSING, resolve_legacy_x_keyword
from .base_detector import BaseDetector
from .baseml import BaseVisionDetector
from .registry import register_model

logger = logging.getLogger(__name__)


@dataclass
class _HSTree:
    """Compact complete binary tree and its two streaming mass profiles."""

    split_dims: np.ndarray
    split_values: np.ndarray
    reference_mass: np.ndarray
    latest_mass: np.ndarray
    max_depth: int


def _build_hst(
    lo: np.ndarray,
    hi: np.ndarray,
    *,
    max_depth: int,
    rng: np.random.Generator,
) -> _HSTree:
    """Build a complete random-dimension, midpoint-cut half-space tree."""

    n_nodes = (1 << (int(max_depth) + 1)) - 1
    split_dims = np.full(n_nodes, -1, dtype=np.int32)
    split_values = np.zeros(n_nodes, dtype=np.float64)

    def build(node: int, depth: int, lower: np.ndarray, upper: np.ndarray) -> None:
        if depth >= int(max_depth):
            return
        dim = int(rng.integers(0, lower.shape[0]))
        cut = float((lower[dim] + upper[dim]) / 2.0)
        split_dims[node] = dim
        split_values[node] = cut

        left_upper = upper.copy()
        left_upper[dim] = cut
        build(2 * node + 1, depth + 1, lower, left_upper)

        right_lower = lower.copy()
        right_lower[dim] = cut
        build(2 * node + 2, depth + 1, right_lower, upper)

    build(0, 0, np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64))
    return _HSTree(
        split_dims=split_dims,
        split_values=split_values,
        reference_mass=np.zeros(n_nodes, dtype=np.int64),
        latest_mass=np.zeros(n_nodes, dtype=np.int64),
        max_depth=int(max_depth),
    )


def _path(tree: _HSTree, x: np.ndarray):  # noqa: ANN202
    """Yield ``(node index, depth)`` along a sample's root-to-leaf path."""

    node = 0
    for depth in range(tree.max_depth + 1):
        yield node, depth
        if depth == tree.max_depth:
            break
        dim = int(tree.split_dims[node])
        if x[dim] < tree.split_values[node]:
            node = 2 * node + 1
        else:
            node = 2 * node + 2


def _update_mass(tree: _HSTree, x: np.ndarray, profile: np.ndarray) -> None:
    for node, _depth in _path(tree, x):
        profile[node] += 1


def _paper_score(tree: _HSTree, x: np.ndarray, *, size_limit: float) -> float:
    """Return the paper score ``reference_mass * 2**depth`` for one tree."""

    for node, depth in _path(tree, x):
        mass = int(tree.reference_mass[node])
        if depth == tree.max_depth or mass <= size_limit:
            return float(mass * (1 << depth))
    raise RuntimeError("unreachable HST scoring state")


@register_model(
    "core_hst",
    tags=("classical", "core", "features", "tree", "online"),
    metadata={
        "description": "Half-Space Trees with reference/latest streaming mass windows",
        "related_paper": "Fast Anomaly Detection for Streaming Data",
        "paper_url": "https://www.ijcai.org/Proceedings/11/Papers/254.pdf",
        "year": 2011,
        "paper_fidelity": "core-aligned",
        "implementation_status": "paper-workspace-and-streaming-mass-protocol",
        "known_deviation": "Returns the reciprocal transform of the paper score so larger values mean more anomalous.",
    },
)
class CoreHST(BaseDetector):
    def __init__(
        self,
        *,
        contamination: float = 0.1,
        n_trees: int = 25,
        max_depth: int = 10,
        window_size: int = 250,
        size_limit: float | None = None,
        assume_normalized: bool = False,
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        super().__init__(contamination=float(contamination))
        self.n_trees = int(n_trees)
        self.max_depth = int(max_depth)
        self.window_size = int(window_size)
        self.size_limit = size_limit
        self.assume_normalized = bool(assume_normalized)
        self.random_state = random_state

    def _validate_parameters(self) -> None:
        if self.n_trees <= 0:
            raise ValueError("n_trees must be > 0")
        if not 1 <= self.max_depth <= 20:
            raise ValueError("max_depth must be between 1 and 20")
        if self.window_size <= 0:
            raise ValueError("window_size must be > 0")
        if self.size_limit is not None and float(self.size_limit) < 0.0:
            raise ValueError("size_limit must be >= 0")

    def _normalize_fit(self, x_arr: np.ndarray) -> np.ndarray:
        if self.assume_normalized:
            self._feature_min = np.zeros(x_arr.shape[1], dtype=np.float64)
            self._feature_scale = np.ones(x_arr.shape[1], dtype=np.float64)
            return x_arr.copy()
        feature_min = np.min(x_arr, axis=0)
        feature_max = np.max(x_arr, axis=0)
        scale = feature_max - feature_min
        scale[scale <= np.finfo(np.float64).eps] = 1.0
        self._feature_min = feature_min
        self._feature_scale = scale
        return (x_arr - feature_min) / scale

    def _normalize_query(self, x_arr: np.ndarray) -> np.ndarray:
        if x_arr.shape[1] != self._feature_min.shape[0]:
            raise ValueError(
                f"X has {x_arr.shape[1]} features, expected {self._feature_min.shape[0]}"
            )
        return (x_arr - self._feature_min) / self._feature_scale

    def _score_normalized(self, x_arr: np.ndarray) -> np.ndarray:
        paper_scores = np.zeros(x_arr.shape[0], dtype=np.float64)
        for tree in self._forest:
            paper_scores += np.asarray(
                [_paper_score(tree, row, size_limit=self._size_limit) for row in x_arr],
                dtype=np.float64,
            )
        paper_scores /= float(len(self._forest))
        return 1.0 / (1.0 + paper_scores)

    def fit(self, x: object = MISSING, y=None, **kwargs: object):  # noqa: ANN001, ANN201
        x_arr = check_array(
            resolve_legacy_x_keyword(x, kwargs, method_name="fit"),
            ensure_2d=True,
            dtype=np.float64,
        )
        self._set_n_classes(y)
        self._validate_parameters()
        if x_arr.shape[0] == 0:
            raise ValueError("X must be non-empty")

        x_norm = self._normalize_fit(x_arr)
        rng = check_random_state(self.random_state)
        forest: list[_HSTree] = []
        for _ in range(self.n_trees):
            # Algorithm 1 creates a data-independent workspace around a random
            # point in the unit hypercube, not around the training observations.
            center = rng.uniform(0.0, 1.0, size=x_norm.shape[1])
            radius = 2.0 * np.maximum(center, 1.0 - center)
            forest.append(
                _build_hst(
                    center - radius,
                    center + radius,
                    max_depth=self.max_depth,
                    rng=rng,
                )
            )

        self._forest = forest
        self._size_limit = (
            0.1 * float(self.window_size) if self.size_limit is None else float(self.size_limit)
        )
        for tree in forest:
            for row in x_norm:
                _update_mass(tree, row, tree.reference_mass)
        self._latest_window_count = 0

        self.decision_scores_ = self._score_normalized(x_norm)
        self._process_decision_scores()
        return self

    def decision_function(self, x: object = MISSING, **kwargs: object):  # noqa: ANN001, ANN201
        require_fitted(self, ["_forest", "_feature_min", "_feature_scale"])
        x_arr = check_array(
            resolve_legacy_x_keyword(x, kwargs, method_name="decision_function"),
            ensure_2d=True,
            dtype=np.float64,
        )
        if x_arr.shape[0] == 0:
            return np.zeros((0,), dtype=np.float64)
        return self._score_normalized(self._normalize_query(x_arr))

    def update(self, x: object = MISSING, **kwargs: object) -> np.ndarray:
        """Score a stream batch, then update/promote the latest mass window."""

        require_fitted(self, ["_forest", "_feature_min", "_feature_scale"])
        x_arr = check_array(
            resolve_legacy_x_keyword(x, kwargs, method_name="update"),
            ensure_2d=True,
            dtype=np.float64,
        )
        if x_arr.shape[0] == 0:
            return np.zeros((0,), dtype=np.float64)
        x_norm = self._normalize_query(x_arr)
        scores = np.zeros(x_norm.shape[0], dtype=np.float64)
        for index, row in enumerate(x_norm):
            scores[index] = self._score_normalized(row[None, :])[0]
            for tree in self._forest:
                _update_mass(tree, row, tree.latest_mass)
            self._latest_window_count += 1
            if self._latest_window_count == self.window_size:
                for tree in self._forest:
                    tree.reference_mass[:] = tree.latest_mass
                    tree.latest_mass.fill(0)
                self._latest_window_count = 0
        return scores


@register_model(
    "vision_hst",
    tags=("vision", "classical", "tree", "online"),
    metadata={
        "description": "Vision wrapper for streaming Half-Space Trees",
        "related_paper": "Fast Anomaly Detection for Streaming Data",
        "paper_url": "https://www.ijcai.org/Proceedings/11/Papers/254.pdf",
        "year": 2011,
        "paper_fidelity": "core-aligned",
        "implementation_status": "vision-wrapper-over-paper-hst-core",
        "known_deviation": "Returns a reciprocal score to satisfy the package's larger-is-more-anomalous contract.",
    },
)
class VisionHST(BaseVisionDetector):
    def __init__(
        self,
        *,
        feature_extractor=None,
        contamination: float = 0.1,
        n_trees: int = 25,
        max_depth: int = 10,
        window_size: int = 250,
        size_limit: float | None = None,
        assume_normalized: bool = False,
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        self._detector_kwargs = {
            "contamination": float(contamination),
            "n_trees": int(n_trees),
            "max_depth": int(max_depth),
            "window_size": int(window_size),
            "size_limit": size_limit,
            "assume_normalized": bool(assume_normalized),
            "random_state": random_state,
        }
        logger.debug("Initializing VisionHST with kwargs=%s", self._detector_kwargs)
        super().__init__(contamination=contamination, feature_extractor=feature_extractor)

    def _build_detector(self):
        return CoreHST(**self._detector_kwargs)
