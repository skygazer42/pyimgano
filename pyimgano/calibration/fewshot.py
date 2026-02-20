from __future__ import annotations

from typing import Literal, Tuple

import numpy as np
from numpy.typing import NDArray

from pyimgano.evaluation import find_optimal_threshold


Objective = Literal["f1", "precision", "recall", "youden"]


def fit_threshold(
    normal_scores: NDArray,
    anomaly_scores: NDArray,
    *,
    objective: Objective = "f1",
) -> float:
    """Fit a decision threshold from few-shot normal/anomalous score samples."""

    normal_scores = np.asarray(normal_scores, dtype=np.float64).ravel()
    anomaly_scores = np.asarray(anomaly_scores, dtype=np.float64).ravel()

    if normal_scores.size == 0:
        raise ValueError("normal_scores cannot be empty")
    if anomaly_scores.size == 0:
        raise ValueError("anomaly_scores cannot be empty")

    y_scores = np.concatenate([normal_scores, anomaly_scores])
    y_true = np.concatenate(
        [np.zeros_like(normal_scores, dtype=np.int64), np.ones_like(anomaly_scores, dtype=np.int64)]
    )

    threshold, _ = find_optimal_threshold(y_true, y_scores, metric=str(objective))
    return float(threshold)


def fit_quantile_threshold(normal_scores: NDArray, *, contamination: float) -> float:
    """Compute a one-class threshold as a quantile of normal scores."""

    if not 0.0 < contamination < 0.5:
        raise ValueError(f"contamination must be in (0, 0.5), got {contamination}")

    normal_scores = np.asarray(normal_scores, dtype=np.float64).ravel()
    if normal_scores.size == 0:
        raise ValueError("normal_scores cannot be empty")

    return float(np.quantile(normal_scores, 1.0 - contamination))


def apply_threshold(scores: NDArray, *, threshold: float) -> NDArray[np.int64]:
    """Convert continuous anomaly scores into 0/1 labels."""

    scores = np.asarray(scores)
    return (scores >= float(threshold)).astype(np.int64)


def split_fewshot(
    scores: NDArray,
    labels: NDArray,
) -> Tuple[NDArray, NDArray]:
    """Split scores into (normal_scores, anomaly_scores) given binary labels."""

    scores = np.asarray(scores)
    labels = np.asarray(labels).astype(np.int64)
    if scores.shape[0] != labels.shape[0]:
        raise ValueError("scores and labels must have the same length")

    normal_scores = scores[labels == 0]
    anomaly_scores = scores[labels == 1]
    return normal_scores, anomaly_scores

