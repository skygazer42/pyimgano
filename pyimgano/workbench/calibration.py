from __future__ import annotations

from typing import Any, Sequence

from pyimgano.inference.api import calibrate_threshold


def _default_quantile(detector: Any, *, fallback: float = 0.995) -> float:
    contamination = getattr(detector, "contamination", None)
    try:
        if contamination is not None:
            cf = float(contamination)
            if 0.0 < cf < 0.5:
                return 1.0 - cf
    except Exception:
        pass
    return float(fallback)


def calibrate_detector_threshold(
    detector: Any,
    calibration_inputs: Sequence[Any],
    *,
    quantile: float | None = None,
    input_format: Any = None,
) -> float:
    q = float(quantile) if quantile is not None else _default_quantile(detector)
    return calibrate_threshold(detector, calibration_inputs, input_format=input_format, quantile=q)

