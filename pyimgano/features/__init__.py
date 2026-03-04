"""Feature extractor utilities for classical detectors."""

from __future__ import annotations

import warnings
from importlib import import_module

from .base import BaseFeatureExtractor
from .identity import IdentityExtractor
from .protocols import FeatureExtractor, FittableFeatureExtractor
from .registry import (
    FEATURE_REGISTRY,
    create_feature_extractor,
    feature_info,
    list_feature_extractors,
    register_feature_extractor,
)

__all__ = [
    "BaseFeatureExtractor",
    "IdentityExtractor",
    "FeatureExtractor",
    "FittableFeatureExtractor",
    "FEATURE_REGISTRY",
    "create_feature_extractor",
    "feature_info",
    "list_feature_extractors",
    "register_feature_extractor",
]


def _auto_import(modules: list[str]) -> None:
    for module_name in modules:
        try:
            import_module(f"{__name__}.{module_name}")
        except Exception as exc:  # noqa: BLE001 - best-effort optional imports
            warnings.warn(
                f"Failed to load feature extractor module {module_name!r}: {exc}",
                RuntimeWarning,
            )


_auto_import(
    [
        "hog",
        "lbp",
        "gabor",
        "color_hist",
        "edge_stats",
        "fft_lowfreq",
        "patch_stats",
        "structural",
        "torchvision_backbone",
        "torchvision_backbone_gem",
        "torchvision_multilayer",
        "torchvision_vit_tokens",
        "torchvision_patch_tokens",
        "patch_grid",
        "torchscript_embed",
        "onnx_embed",
        "openclip_embed",
        "multi",
        "pca_projector",
        "scaler",
        "normalize",
    ]
)
