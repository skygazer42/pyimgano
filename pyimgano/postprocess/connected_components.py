from __future__ import annotations

"""Connected-components utilities for anomaly masks/maps.

This is used in industrial post-processing to:
- remove tiny spurious detections
- score and rank defect regions
"""

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np


_Reducer = Literal["max", "mean", "sum"]


@dataclass(frozen=True)
class Component:
    label: int
    area: int
    bbox_xywh: tuple[int, int, int, int]
    centroid_xy: tuple[float, float]
    score: float | None = None


def label_components(binary_mask_u8: np.ndarray, *, connectivity: int = 8) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Return (num_labels, labels, stats, centroids) using OpenCV labeling.

    `binary_mask_u8` is treated as boolean (non-zero = foreground).
    """

    import cv2  # local import

    m = np.asarray(binary_mask_u8)
    if m.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {m.shape}")
    # cv2 expects uint8 0/1-ish.
    m_u8 = (m > 0).astype(np.uint8, copy=False)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        m_u8, connectivity=int(connectivity)
    )
    return int(num_labels), np.asarray(labels), np.asarray(stats), np.asarray(centroids)


def components_from_labels(
    num_labels: int,
    *,
    labels: np.ndarray,
    stats: np.ndarray,
    centroids: np.ndarray,
    scores: Sequence[float] | None = None,
) -> list[Component]:
    """Build a list of Component entries (excluding background label 0)."""

    out: list[Component] = []
    for lab in range(1, int(num_labels)):
        x = int(stats[lab, 0])
        y = int(stats[lab, 1])
        w = int(stats[lab, 2])
        h = int(stats[lab, 3])
        area = int(stats[lab, 4])
        cx = float(centroids[lab, 0])
        cy = float(centroids[lab, 1])
        score = None if scores is None else float(scores[lab])
        out.append(
            Component(
                label=int(lab),
                area=int(area),
                bbox_xywh=(x, y, w, h),
                centroid_xy=(cx, cy),
                score=score,
            )
        )
    return out


def score_components(
    anomaly_map: np.ndarray,
    *,
    labels: np.ndarray,
    num_labels: int,
    reducer: _Reducer = "max",
) -> np.ndarray:
    """Compute per-component score from an anomaly map and component labels.

    Returns a float array of length `num_labels` where index 0 (background)
    is always 0.
    """

    m = np.asarray(anomaly_map, dtype=np.float32)
    if m.ndim != 2:
        raise ValueError(f"Expected 2D anomaly map, got shape {m.shape}")
    lab = np.asarray(labels)
    if lab.shape != m.shape:
        raise ValueError(f"labels shape must match anomaly_map. Got {lab.shape} vs {m.shape}")

    scores = np.zeros((int(num_labels),), dtype=np.float64)
    if int(num_labels) <= 1:
        return scores

    for i in range(1, int(num_labels)):
        region = m[lab == i]
        if region.size == 0:
            scores[i] = 0.0
        elif reducer == "max":
            scores[i] = float(np.max(region))
        elif reducer == "mean":
            scores[i] = float(np.mean(region))
        elif reducer == "sum":
            scores[i] = float(np.sum(region))
        else:  # pragma: no cover - guarded by Literal type
            raise ValueError(f"Unknown reducer: {reducer!r}")

    return scores


def filter_small_components(
    anomaly_map: np.ndarray,
    *,
    threshold: float,
    min_area: int,
    connectivity: int = 8,
    reducer: _Reducer = "max",
) -> tuple[np.ndarray, list[Component]]:
    """Zero-out components smaller than `min_area` and return kept component info.

    Returns
    -------
    filtered_map:
        Same shape as input, float32.
    kept_components:
        Sorted by descending score (then area).
    """

    m = np.asarray(anomaly_map, dtype=np.float32)
    if m.ndim != 2:
        raise ValueError(f"Expected 2D anomaly map, got shape {m.shape}")

    binary = (m >= float(threshold)).astype(np.uint8)
    num, labels, stats, centroids = label_components(binary, connectivity=int(connectivity))
    if int(num) <= 1:
        return m, []

    scores = score_components(m, labels=labels, num_labels=int(num), reducer=reducer)

    keep = np.zeros((int(num),), dtype=bool)
    for lab in range(1, int(num)):
        area = int(stats[lab, 4])
        keep[lab] = area >= int(min_area)

    mask_keep = keep[labels]
    out = m.copy()
    out[~mask_keep] = 0.0

    comps = components_from_labels(
        int(num),
        labels=labels,
        stats=stats,
        centroids=centroids,
        scores=scores,
    )
    comps = [c for c in comps if c.area >= int(min_area)]
    comps.sort(key=lambda c: (-(c.score or 0.0), -int(c.area), int(c.label)))
    return out, comps


__all__ = [
    "Component",
    "label_components",
    "score_components",
    "filter_small_components",
]

