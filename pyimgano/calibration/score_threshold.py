from __future__ import annotations

from typing import Any, Literal

QuantileSource = Literal["explicit", "contamination", "fallback"]


def resolve_calibration_quantile(
    detector: Any,
    *,
    calibration_quantile: float | None,
    fallback: float = 0.995,
) -> tuple[float, QuantileSource]:
    """Resolve a score calibration quantile for a detector.

    Resolution order (first match wins):

    1) Explicit `calibration_quantile` provided by the caller.
    2) `1 - detector.contamination` when contamination is in (0, 0.5).
    3) `fallback` (default: 0.995).

    Returns:
        (quantile, source) where source is one of: explicit, contamination, fallback.
    """

    if calibration_quantile is not None:
        q = float(calibration_quantile)
        if not 0.0 < q < 1.0:
            raise ValueError(f"calibration_quantile must be in (0,1), got {q}")
        return q, "explicit"

    contamination = getattr(detector, "contamination", None)
    try:
        if contamination is not None:
            cf = float(contamination)
            if 0.0 < cf < 0.5:
                return 1.0 - cf, "contamination"
    except Exception:
        pass

    q = float(fallback)
    if not 0.0 < q < 1.0:
        raise ValueError(f"fallback quantile must be in (0,1), got {q}")
    return q, "fallback"

