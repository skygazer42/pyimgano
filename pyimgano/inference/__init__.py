"""Industrial inference helpers (numpy-first).

This module provides a small, production-friendly API on top of detectors:
- explicit input formats (`ImageFormat`)
- optional quantile threshold calibration
- structured per-image inference results
"""

from __future__ import annotations

from .api import (
    InferenceResult,
    InferenceTiming,
    calibrate_threshold,
    calibrate_threshold_bgr,
    infer,
    infer_bgr,
    infer_iter,
    infer_iter_bgr,
    result_to_jsonable,
    results_to_jsonable,
)
from .tiling import TiledDetector

__all__ = [
    "InferenceTiming",
    "InferenceResult",
    "calibrate_threshold_bgr",
    "calibrate_threshold",
    "infer",
    "infer_bgr",
    "infer_iter",
    "infer_iter_bgr",
    "result_to_jsonable",
    "results_to_jsonable",
    "TiledDetector",
]
