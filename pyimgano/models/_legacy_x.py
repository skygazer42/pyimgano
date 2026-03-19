"""Helpers for keeping legacy `X=` keyword compatibility.

Some detectors historically exposed methods like `predict(X=...)` and
`decision_function(X=...)`. We now prefer lowercase local naming to satisfy
linting, but still accept the legacy keyword to avoid breaking callers.
"""

from __future__ import annotations

from typing import Dict, TypeVar, cast

T = TypeVar("T")

MISSING = object()


def resolve_legacy_x_keyword(x: object, kwargs: Dict[str, object], *, method_name: str) -> T:
    """Return the effective input after accepting legacy `X=`."""

    legacy_x = kwargs.pop("X", MISSING)
    if kwargs:
        unexpected = next(iter(kwargs))
        raise TypeError(f"{method_name}() got an unexpected keyword argument {unexpected!r}")
    if x is MISSING:
        if legacy_x is MISSING:
            raise TypeError(f"{method_name}() missing 1 required positional argument: 'x'")
        return cast(T, legacy_x)
    if legacy_x is not MISSING:
        raise TypeError(f"{method_name}() got multiple values for argument 'x'")
    return cast(T, x)
