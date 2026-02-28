from __future__ import annotations

"""Reference-based pixel anomaly map pipeline (query vs golden reference).

Industrial settings like RAD/ReinAD often provide a "golden" reference image
for each inspected sample. This module provides a small, dependency-light base
pipeline that:

- maps each query image path -> reference image path (by filename by default)
- computes an anomaly map from (query, reference)
- reduces the anomaly map into an image-level score (max/mean/topk_mean)

This is intended as a base class for reference-based pixel-map detectors.
"""

from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np

from pyimgano.models.base_detector import BaseDetector


_Reduce = Literal["max", "mean", "topk_mean"]
_MatchMode = Literal["basename"]


def _iter_image_paths(root: Path) -> list[Path]:
    suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    out: list[Path] = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in suffixes:
            out.append(p)
    return out


def build_reference_index(reference_dir: str | Path, *, match_mode: _MatchMode = "basename") -> dict[str, str]:
    """Build a mapping key -> absolute reference path.

    match_mode="basename" maps by filename (e.g. "IMG_001.png").

    Raises
    ------
    ValueError
        If the index would be ambiguous (duplicate keys).
    """

    mode = str(match_mode).strip().lower()
    if mode != "basename":
        raise ValueError("match_mode must be 'basename' (v1)")

    root = Path(reference_dir)
    if not root.is_dir():
        raise ValueError(f"reference_dir must be a directory, got: {root}")

    index: dict[str, str] = {}
    duplicates: dict[str, list[str]] = {}
    for p in _iter_image_paths(root):
        key = p.name
        resolved = str(p.resolve())
        if key in index and index[key] != resolved:
            duplicates.setdefault(key, [index[key]])
            duplicates[key].append(resolved)
            continue
        index[key] = resolved

    if duplicates:
        example = sorted(duplicates.items())[0]
        key, paths = example
        raise ValueError(
            "Reference directory contains ambiguous duplicate filenames. "
            f"Duplicate key={key!r} example_paths={paths[:3]!r}. "
            "Fix by ensuring unique basenames under --reference-dir."
        )

    if not index:
        raise ValueError(f"No reference images found under: {root}")

    return index


def _topk_mean(values: np.ndarray, *, topk: float) -> float:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return 0.0
    frac = float(topk)
    if not (0.0 < frac <= 1.0):
        raise ValueError("topk must be in (0,1].")
    k = max(1, int(np.ceil(frac * float(arr.size))))
    k = min(k, int(arr.size))
    top_vals = np.partition(arr, -k)[-k:]
    return float(np.mean(top_vals))


@dataclass
class ReferenceMapPipeline(BaseDetector):
    """Base detector for reference-based pixel anomaly maps.

    Subclasses implement `_compute_anomaly_map(query_path, reference_path)`.
    """

    contamination: float = 0.1
    reference_dir: str | Path | None = None
    match_mode: _MatchMode = "basename"
    reduction: _Reduce = "max"
    topk: float = 0.1

    def __post_init__(self) -> None:
        super().__init__(contamination=float(self.contamination))
        self._reference_index: dict[str, str] | None = None
        if self.reference_dir is not None:
            self.set_reference_dir(self.reference_dir)

        red = str(self.reduction).strip().lower()
        if red not in ("max", "mean", "topk_mean"):
            raise ValueError("reduction must be one of: max|mean|topk_mean")
        self.reduction = red  # type: ignore[assignment]

    # ------------------------------------------------------------------
    def set_reference_dir(self, reference_dir: str | Path) -> None:
        """Set the reference directory and build an index for fast lookup."""

        self.reference_dir = reference_dir
        self._reference_index = build_reference_index(reference_dir, match_mode=self.match_mode)

    def _resolve_reference_path(self, query_path: str | Path) -> str:
        if self._reference_index is None:
            raise ValueError("reference_dir is not set. Provide --reference-dir or call set_reference_dir().")
        key = Path(str(query_path)).name
        ref = self._reference_index.get(key, None)
        if ref is None:
            raise FileNotFoundError(
                f"No reference found for query={str(query_path)!r} (key={key!r}). "
                f"reference_dir={str(self.reference_dir)!r}."
            )
        return str(ref)

    # ------------------------------------------------------------------
    @abstractmethod
    def _compute_anomaly_map(self, *, query_path: str, reference_path: str) -> np.ndarray:
        """Return anomaly map for a single (query, reference) pair (H,W float32)."""

        raise NotImplementedError

    def get_anomaly_map(self, item: Any) -> np.ndarray:
        if not isinstance(item, (str, Path)):
            raise TypeError("Reference-based pipelines currently require path inputs.")

        q = str(item)
        r = self._resolve_reference_path(q)
        amap = self._compute_anomaly_map(query_path=q, reference_path=r)
        arr = np.asarray(amap, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"_compute_anomaly_map must return a 2D array, got shape {arr.shape}")
        return arr

    def _score_from_map(self, anomaly_map: np.ndarray) -> float:
        arr = np.asarray(anomaly_map, dtype=np.float32)
        if self.reduction == "max":
            return float(np.max(arr)) if arr.size else 0.0
        if self.reduction == "mean":
            return float(np.mean(arr)) if arr.size else 0.0
        if self.reduction == "topk_mean":
            return _topk_mean(arr, topk=float(self.topk))
        raise ValueError("Unknown reduction mode")

    # ------------------------------------------------------------------
    def fit(self, X: Iterable[Any], y=None):  # noqa: ANN001, ANN201 - sklearn-like API
        items = list(X)
        if not items:
            raise ValueError("Training set cannot be empty")

        self._set_n_classes(y)
        self.decision_scores_ = np.asarray(self.decision_function(items), dtype=np.float64).reshape(-1)
        self._process_decision_scores()
        return self

    def decision_function(self, X: Iterable[Any]):  # noqa: ANN001, ANN201 - sklearn-like API
        items = list(X)
        scores: list[float] = []
        for it in items:
            amap = self.get_anomaly_map(it)
            scores.append(float(self._score_from_map(amap)))
        return np.asarray(scores, dtype=np.float64).reshape(-1)


__all__ = ["ReferenceMapPipeline", "build_reference_index"]

