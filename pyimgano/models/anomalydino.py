from __future__ import annotations

import math
from typing import Literal

import numpy as np
from numpy.typing import NDArray


_AggregationMethod = Literal["topk_mean", "max", "mean"]


def _aggregate_patch_scores(
    patch_scores: NDArray,
    *,
    method: _AggregationMethod = "topk_mean",
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

        k = max(1, int(math.ceil(topk_float * scores.size)))
        k = min(k, scores.size)

        top_scores = np.partition(scores, -k)[-k:]
        return float(np.mean(top_scores))

    raise ValueError(f"Unknown aggregation method: {method}. Choose from: topk_mean, max, mean")


def _reshape_patch_scores(
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

