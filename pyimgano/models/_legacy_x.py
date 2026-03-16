from __future__ import annotations

from typing import Any

MISSING = object()


def resolve_legacy_x_keyword(x: object, kwargs: dict[str, Any], *, method_name: str) -> object:
    """Accept legacy `X=` inputs while standardizing on lowercase `x`."""

    legacy_x = kwargs.pop("X", MISSING)
    if kwargs:
        unexpected = next(iter(kwargs))
        raise TypeError(f"{method_name}() got an unexpected keyword argument {unexpected!r}")
    if x is MISSING:
        if legacy_x is MISSING:
            raise TypeError(f"{method_name}() missing 1 required positional argument: 'x'")
        return legacy_x
    if legacy_x is not MISSING:
        raise TypeError(f"{method_name}() got multiple values for argument 'x'")
    return x
