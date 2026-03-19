from __future__ import annotations

"""Shared helpers for deep/vision model contracts.

This module does not implement detectors by itself. It provides small utilities
to help deep models conform to the repository-wide conventions:

- `decision_function(X)` returns a 1D score array
- higher score = more anomalous
- scores are numpy float64 (or convertible)
- optional pixel-level outputs are exposed via `get_anomaly_map(...)` when a
  model supports it
"""

from dataclasses import dataclass
from typing import Any, Optional, Protocol, runtime_checkable

import numpy as np

from pyimgano.models.protocols import normalize_anomaly_maps, normalize_scores


@runtime_checkable
class SupportsAnomalyMap(Protocol):
    def get_anomaly_map(self, x: Any) -> np.ndarray:  # noqa: ANN401 - protocol
        """Return an anomaly map for a single image (H,W) or batch (N,H,W)."""


def ensure_scores_1d(scores: Any, *, n: int | None = None) -> np.ndarray:
    """Convert model scores to a 1D float64 numpy array and validate length."""

    arr = normalize_scores(scores, n_expected=None if n is None else int(n))
    return np.asarray(arr, dtype=np.float64)


def ensure_anomaly_map_2d(anomaly_map: Any, *, hw: Optional[tuple[int, int]] = None) -> np.ndarray:
    """Convert an anomaly map to float32 (H,W) and optionally validate size."""

    m = normalize_anomaly_maps(anomaly_map, n_expected=1)[0]
    if hw is not None:
        h, w = map(int, hw)
        if m.shape != (h, w):
            raise ValueError(f"Anomaly map shape mismatch: expected {(h, w)}, got {m.shape}")
    return m


@dataclass(frozen=True)
class DeepIOState:
    """Minimal checkpoint payload for deep detectors."""

    state_dict: dict[str, Any]
    meta: dict[str, Any]


__all__ = [
    "SupportsAnomalyMap",
    "ensure_scores_1d",
    "ensure_anomaly_map_2d",
    "DeepIOState",
]
