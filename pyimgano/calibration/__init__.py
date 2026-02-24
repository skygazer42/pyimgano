"""Calibration helpers (thresholds, quantiles, few-shot).

Keep imports lightweight: some calibration helpers depend on heavier optional
dependencies (e.g. scikit-learn). We lazy-load the public exports on demand so
`import pyimgano.calibration` stays usable in minimal environments.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["fit_threshold", "calibrate_normal_pixel_quantile_threshold", "resolve_calibration_quantile"]

_LAZY_EXPORTS = {
    "fit_threshold": ("fewshot", "fit_threshold"),
    "calibrate_normal_pixel_quantile_threshold": (
        "pixel_threshold",
        "calibrate_normal_pixel_quantile_threshold",
    ),
    "resolve_calibration_quantile": ("score_threshold", "resolve_calibration_quantile"),
}


def __getattr__(name: str) -> Any:  # pragma: no cover - thin lazy import
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr = target
    module = import_module(f"{__name__}.{module_name}")
    value = getattr(module, attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:  # pragma: no cover - tooling convenience
    return sorted(set(globals()) | set(__all__))
