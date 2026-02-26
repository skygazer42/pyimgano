"""Shared typing helpers.

This module exists to avoid repeating verbose `numpy.typing` types and to keep
cross-module type hints consistent.
"""

from __future__ import annotations

from typing import Iterable, Sequence, Union

import numpy as np
from numpy.typing import NDArray


ArrayLike = Union[Sequence[float], NDArray[np.generic]]
FloatArray = NDArray[np.floating]
IntArray = NDArray[np.integer]
UInt8Array = NDArray[np.uint8]


PathLikeStr = Union[str, "os.PathLike[str]"]  # pragma: no cover - typing only


def ensure_1d_float_array(x) -> FloatArray:
    """Best-effort conversion to 1D float numpy array."""

    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    return arr  # type: ignore[return-value]


def ensure_2d_float_array(x) -> FloatArray:
    """Best-effort conversion to 2D float numpy array."""

    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {arr.shape}")
    return arr  # type: ignore[return-value]


def ensure_iterable_str(x) -> Iterable[str]:
    """Coerce input into an iterable of strings (paths/ids)."""

    if isinstance(x, (str, bytes)):
        return [str(x)]
    return [str(v) for v in x]

