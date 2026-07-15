from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

AggregationMethod = Literal["topk_mean", "max", "mean"]


def approximate_greedy_coreset_indices(
    features: NDArray,
    *,
    sampling_ratio: float,
    projection_dim: int | None = 128,
    starting_points: int = 10,
    random_seed: int = 0,
    sample_count: int | None = None,
) -> NDArray[np.int64]:
    """Select PatchCore-style approximate greedy k-center indices."""

    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] < 1:
        raise ValueError(f"features must be a non-empty 2D matrix, got {values.shape}")
    if not 0.0 < float(sampling_ratio) <= 1.0:
        raise ValueError(f"sampling_ratio must be in (0, 1], got {sampling_ratio}")
    if projection_dim is not None and int(projection_dim) < 1:
        raise ValueError(f"projection_dim must be positive or None, got {projection_dim}")
    if int(starting_points) < 1:
        raise ValueError(f"starting_points must be positive, got {starting_points}")

    n_samples = int(values.shape[0])
    if sample_count is not None and int(sample_count) < 1:
        raise ValueError(f"sample_count must be positive or None, got {sample_count}")
    n_select = (
        max(1, int(n_samples * float(sampling_ratio)))
        if sample_count is None
        else min(int(sample_count), n_samples)
    )
    if n_select >= n_samples:
        return np.arange(n_samples, dtype=np.int64)

    rng = np.random.default_rng(int(random_seed))
    selection = values
    if projection_dim is not None and selection.shape[1] != int(projection_dim):
        projection = rng.standard_normal((selection.shape[1], int(projection_dim))).astype(
            np.float32
        )
        projection /= np.sqrt(float(projection_dim))
        selection = selection @ projection

    squared_norms = np.sum(selection * selection, axis=1)

    def distances_to(indices: NDArray) -> NDArray:
        anchors = selection[np.asarray(indices, dtype=np.int64)]
        distances_sq = (
            squared_norms[:, None]
            + np.sum(anchors * anchors, axis=1)[None, :]
            - 2.0 * selection @ anchors.T
        )
        return np.sqrt(np.maximum(distances_sq, 0.0))

    anchors = rng.choice(
        n_samples,
        size=min(int(starting_points), n_samples),
        replace=False,
    )
    anchor_distances = distances_to(anchors).mean(axis=1)
    selected: list[int] = []
    for _ in range(n_select):
        next_idx = int(np.argmax(anchor_distances))
        selected.append(next_idx)
        anchor_distances = np.minimum(
            anchor_distances,
            distances_to(np.asarray([next_idx], dtype=np.int64)).reshape(-1),
        )
    return np.asarray(selected, dtype=np.int64)


def aggregate_patch_scores(
    patch_scores: NDArray,
    *,
    method: AggregationMethod = "topk_mean",
    topk: float = 0.01,
) -> float:
    scores = np.asarray(patch_scores, dtype=np.float64).ravel()
    if scores.size == 0:
        raise ValueError("patch_scores must be non-empty")

    method_lower = str(method).lower()
    if method_lower == "max":
        return float(np.max(scores))
    if method_lower == "mean":
        return float(np.mean(scores))
    if method_lower == "topk_mean":
        topk_float = float(topk)
        if not (0.0 < topk_float <= 1.0):
            raise ValueError("topk must be a fraction in (0, 1].")

        k = max(1, int(topk_float * scores.size))
        k = min(k, scores.size)

        top_scores = np.partition(scores, -k)[-k:]
        return float(np.mean(top_scores))

    raise ValueError(f"Unknown aggregation method: {method}. Choose from: topk_mean, max, mean")


def reshape_patch_scores(
    patch_scores: NDArray,
    *,
    grid_h: int,
    grid_w: int,
) -> NDArray:
    scores = np.asarray(patch_scores)
    if scores.ndim != 1:
        scores = scores.reshape(-1)

    grid_h_int = int(grid_h)
    grid_w_int = int(grid_w)
    expected = grid_h_int * grid_w_int
    if scores.size != expected:
        raise ValueError(
            f"Expected {expected} patch scores for grid {grid_h_int}x{grid_w_int}, got {scores.size}."
        )

    return scores.reshape(grid_h_int, grid_w_int)
