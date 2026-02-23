from __future__ import annotations

from pathlib import Path
from typing import Any

from pyimgano.utils.optional_deps import optional_import

_NUMPY, _NUMPY_ERROR = optional_import("numpy")


def to_jsonable(value: Any) -> Any:
    """Convert common scientific/python objects into JSON-serializable values.

    - `pathlib.Path` → `str`
    - `numpy` scalars → builtin Python scalars via `.item()`
    - `numpy.ndarray` → nested Python lists via `.tolist()`
    - Recurses through `dict` / `list` / `tuple`

    Notes
    -----
    This function is intentionally dependency-light: NumPy is optional at import
    time so CLIs can remain importable in minimal environments.
    """

    if isinstance(value, Path):
        return str(value)

    if _NUMPY is not None:
        if isinstance(value, _NUMPY.generic):  # type: ignore[attr-defined]
            return value.item()
        if isinstance(value, _NUMPY.ndarray):  # type: ignore[attr-defined]
            return value.tolist()

    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value

