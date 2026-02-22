"""Industrial inference helpers (numpy-first).

This module provides a small, production-friendly API on top of detectors:
- explicit input formats (`ImageFormat`)
- optional quantile threshold calibration
- structured per-image inference results
"""

from __future__ import annotations

from .api import (
    InferenceResult,
    calibrate_threshold,
    infer,
    result_to_jsonable,
    results_to_jsonable,
)

__all__ = [
    "InferenceResult",
    "calibrate_threshold",
    "infer",
    "result_to_jsonable",
    "results_to_jsonable",
]
