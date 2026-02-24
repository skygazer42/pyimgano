from __future__ import annotations

from typing import Any, Sequence

from pyimgano.inference.api import calibrate_threshold


def resolve_default_quantile(detector: Any, *, fallback: float = 0.995) -> tuple[float, str]:
    """Resolve the default threshold calibration quantile for a detector.

    Returns:
        (quantile, source) where source is one of: "contamination", "fallback".

    Notes:
        This helper intentionally does not accept an explicit override; callers
        that need an explicit quantile should call
        `pyimgano.calibration.score_threshold.resolve_calibration_quantile`.
    """

    from pyimgano.calibration.score_threshold import resolve_calibration_quantile

    q, src = resolve_calibration_quantile(detector, calibration_quantile=None, fallback=fallback)
    if src == "explicit":  # pragma: no cover - guarded by calibration_quantile=None
        src = "fallback"
    return float(q), str(src)


def _default_quantile(detector: Any, *, fallback: float = 0.995) -> float:
    q, _src = resolve_default_quantile(detector, fallback=fallback)
    return float(q)


def calibrate_detector_threshold(
    detector: Any,
    calibration_inputs: Sequence[Any],
    *,
    quantile: float | None = None,
    input_format: Any = None,
) -> float:
    q = float(quantile) if quantile is not None else _default_quantile(detector)
    return calibrate_threshold(detector, calibration_inputs, input_format=input_format, quantile=q)
