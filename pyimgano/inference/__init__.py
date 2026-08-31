"""Industrial inference helpers (numpy-first).

This module provides a small, production-friendly API on top of detectors:
- explicit input formats (`ImageFormat`)
- optional quantile threshold calibration
- structured per-image inference results
"""

from __future__ import annotations

from typing import Any

from .api import (
    INFERENCE_UNSET,
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
from .artifact_runtime import ArtifactRuntime, ArtifactRuntimeError
from .legacy_artifact import LegacyArtifactWarning, load_legacy_artifact
from .tiling import TiledDetector


def __getattr__(name: str) -> Any:
    """Load the service facade lazily to keep fresh submodule imports acyclic."""

    if name == "load_artifact":
        from pyimgano.services.artifact_load_service import load_artifact

        globals()[name] = load_artifact
        return load_artifact
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "InferenceTiming",
    "InferenceResult",
    "INFERENCE_UNSET",
    "ArtifactRuntime",
    "ArtifactRuntimeError",
    "LegacyArtifactWarning",
    "calibrate_threshold_bgr",
    "calibrate_threshold",
    "infer",
    "infer_bgr",
    "infer_iter",
    "infer_iter_bgr",
    "result_to_jsonable",
    "results_to_jsonable",
    "load_artifact",
    "load_legacy_artifact",
    "TiledDetector",
]
