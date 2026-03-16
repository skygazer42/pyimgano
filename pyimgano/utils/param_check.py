"""Small parameter validation helpers.

We keep these utilities minimal (no extra deps) and stable because they are
used across multiple detectors.
"""

from __future__ import annotations

from numbers import Number
from typing import cast


def _validate_bound(value: Number | None, *, bound_name: str) -> Number | None:
    if value is None:
        return None
    if not isinstance(value, Number) or isinstance(value, bool):
        raise TypeError(f"{bound_name} must be a number, got {type(value).__name__}")
    return cast(Number, value)


def check_parameter(
    param: Number,
    low: Number | None = None,
    high: Number | None = None,
    *,
    param_name: str = "parameter",
    include_left: bool = True,
    include_right: bool = True,
) -> None:
    """Validate a numeric parameter is within a given range.

    Parameters
    ----------
    param:
        The numeric value to validate.
    low / high:
        Optional bounds. When `None`, the bound is not checked.
    include_left / include_right:
        Whether the comparison is inclusive.
    param_name:
        Used in error messages.
    """

    if not isinstance(param, Number) or isinstance(param, bool):
        raise TypeError(f"{param_name} must be a number, got {type(param).__name__}")

    low_value = _validate_bound(low, bound_name="low")
    high_value = _validate_bound(high, bound_name="high")

    if low_value is not None and high_value is not None and low_value > high_value:
        raise ValueError(f"Invalid bounds for {param_name}: low={low} > high={high}")

    if low_value is not None:
        if include_left:
            if param < low_value:
                raise ValueError(f"{param_name} must be >= {low_value}, got {param}")
        else:
            if param <= low_value:
                raise ValueError(f"{param_name} must be > {low_value}, got {param}")

    if high_value is not None:
        if include_right:
            if param > high_value:
                raise ValueError(f"{param_name} must be <= {high_value}, got {param}")
        else:
            if param >= high_value:
                raise ValueError(f"{param_name} must be < {high_value}, got {param}")
