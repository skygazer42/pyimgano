"""Shared helpers for detector implementations.

We keep this module small and dependency-stable. It exists to reduce copy/paste
validation logic across detectors.
"""

from __future__ import annotations

from typing import Any

from pyimgano.utils.param_check import check_parameter


def ensure_int(value: Any, *, name: str, low: int | None = None, high: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, (int,)):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    v = int(value)
    if low is not None and v < int(low):
        raise ValueError(f"{name} must be >= {low}, got {v}")
    if high is not None and v > int(high):
        raise ValueError(f"{name} must be <= {high}, got {v}")
    return v


def ensure_contamination(value: Any) -> float:
    c = float(value)
    check_parameter(
        c, low=0.0, high=0.5, include_left=False, include_right=True, param_name="contamination"
    )
    return c
