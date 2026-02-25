"""Industrial inference helpers (numpy-first).

This module provides a small, production-friendly API on top of detectors:
- explicit input formats (`ImageFormat`)
- optional quantile threshold calibration
- structured per-image inference results
"""

from __future__ import annotations

from .api import (
    InferenceTiming,
    InferenceResult,
    calibrate_threshold,
    infer,
    infer_iter,
    result_to_jsonable,
    results_to_jsonable,
)
from .tiling import TiledDetector

__all__ = [
    "InferenceTiming",
    "InferenceResult",
    "calibrate_threshold",
    "infer",
    "infer_iter",
    "result_to_jsonable",
    "results_to_jsonable",
    "TiledDetector",
]
