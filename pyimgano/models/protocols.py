from __future__ import annotations

from typing import Any, Iterable, Literal, Protocol, runtime_checkable

import numpy as np

InputMode = Literal["paths", "numpy", "features"]


@runtime_checkable
class DetectorProtocol(Protocol):
    input_mode: InputMode

    def fit(self, X: Iterable[Any], y: Any | None = None) -> "DetectorProtocol":
        ...

    def decision_function(self, X: Iterable[Any]) -> np.ndarray:
        ...

    def predict(self, X: Iterable[Any]) -> np.ndarray:
        ...


@runtime_checkable
class PixelMapDetectorProtocol(DetectorProtocol, Protocol):
    def predict_anomaly_map(self, X: Iterable[Any]) -> Any:
        ...


def normalize_scores(scores: Any, *, n_expected: int | None = None) -> np.ndarray:
    arr = np.asarray(scores, dtype=np.float32)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    if n_expected is not None and int(arr.shape[0]) != int(n_expected):
        raise ValueError(
            f"Expected {int(n_expected)} scores, got array with shape {tuple(arr.shape)}."
        )
    return arr


def normalize_single_anomaly_map(anomaly_map: Any) -> np.ndarray:
    arr = np.asarray(anomaly_map, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected anomaly map with shape (H, W), got {tuple(arr.shape)}.")
    return arr


def normalize_anomaly_maps(maps: Any, *, n_expected: int) -> np.ndarray:
    if isinstance(maps, (list, tuple)):
        if not maps:
            raise ValueError("Expected one or more anomaly maps, got an empty list.")
        arr = np.stack([normalize_single_anomaly_map(m) for m in maps], axis=0)
    else:
        arr = np.asarray(maps, dtype=np.float32)
        if arr.ndim == 2 and int(n_expected) == 1:
            arr = arr[None, ...]

    if arr.ndim != 3:
        raise ValueError(
            f"Expected anomaly maps with shape (N, H, W), got {tuple(np.asarray(arr).shape)}."
        )
    if int(arr.shape[0]) != int(n_expected):
        raise ValueError(
            f"Expected {int(n_expected)} anomaly maps, got array with shape {tuple(arr.shape)}."
        )
    return np.asarray(arr, dtype=np.float32)


__all__ = [
    "DetectorProtocol",
    "InputMode",
    "PixelMapDetectorProtocol",
    "normalize_anomaly_maps",
    "normalize_scores",
    "normalize_single_anomaly_map",
]
