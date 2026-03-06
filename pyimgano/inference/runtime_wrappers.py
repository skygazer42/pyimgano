from __future__ import annotations

from typing import Any

_RUNTIME_WRAPPER_TYPES = {
    "pyimgano.inference.preprocessing.PreprocessingDetector",
    "pyimgano.inference.tiling.TiledDetector",
}


def is_runtime_detector_wrapper(obj: Any) -> bool:
    cls = type(obj)
    qualname = f"{cls.__module__}.{cls.__name__}"
    return qualname in _RUNTIME_WRAPPER_TYPES


def unwrap_runtime_detector(obj: Any) -> Any:
    current = obj
    seen: set[int] = set()

    while is_runtime_detector_wrapper(current):
        current_id = id(current)
        if current_id in seen:
            break
        seen.add(current_id)

        try:
            current = object.__getattribute__(current, "detector")
        except AttributeError:
            break

    return current
