"""Small parameter validation helpers.

We keep these utilities minimal (no extra deps) and stable because they are
used across multiple detectors.
"""

from __future__ import annotations

from numbers import Number


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

    if low is not None and high is not None and low > high:
        raise ValueError(f"Invalid bounds for {param_name}: low={low} > high={high}")

    if low is not None:
        if include_left:
            if param < low:
                raise ValueError(f"{param_name} must be >= {low}, got {param}")
        else:
            if param <= low:
                raise ValueError(f"{param_name} must be > {low}, got {param}")

    if high is not None:
        if include_right:
            if param > high:
                raise ValueError(f"{param_name} must be <= {high}, got {param}")
        else:
            if param >= high:
                raise ValueError(f"{param_name} must be < {high}, got {param}")

