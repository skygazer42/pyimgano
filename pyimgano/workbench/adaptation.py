"""Compatibility facade for workbench adaptation types and runtime helpers."""

from __future__ import annotations

from pyimgano.workbench.adaptation_runtime import apply_tiling, build_postprocess
from pyimgano.workbench.adaptation_types import AdaptationConfig, MapPostprocessConfig, TilingConfig

__all__ = [
    "TilingConfig",
    "MapPostprocessConfig",
    "AdaptationConfig",
    "apply_tiling",
    "build_postprocess",
]
