"""
Datasets module providing image data loading and preprocessing components.

This module includes dataset classes, data loaders, and transformation utilities
for vision-based anomaly detection tasks.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .benchmarks import (
    BTADDataset,
    BaseDataset,
    CustomDataset,
    DatasetInfo,
    ManifestDataset,
    MVTecAD2Dataset,
    MVTecDataset,
    MVTecLOCODataset,
    VisADataset,
    load_dataset,
)

__all__ = [
    # Benchmark datasets (industrial evaluation)
    "BaseDataset",
    "DatasetInfo",
    "MVTecDataset",
    "MVTecLOCODataset",
    "MVTecAD2Dataset",
    "VisADataset",
    "BTADDataset",
    "CustomDataset",
    "ManifestDataset",
    "load_dataset",
    # Dataset → manifest converters (paths-first)
    "convert_mvtec_ad2_to_manifest",
    "convert_real_iad_to_manifest",
    "convert_rad_to_manifest",
    # Torch-style datasets / datamodule
    "ImagePathDataset",
    "VisionImageDataset",
    "VisionArrayDataset",
    "default_eval_transforms",
    "default_train_transforms",
    "to_tensor_normalized",
    "DataLoaderConfig",
    "VisionDataModule",
    # Synthetic anomalies
    "SyntheticAnomalyDataset",
    "SyntheticItem",
    # Robustness / corruptions
    "CorruptionsDataset",
    "CorruptionItem",
]


_LAZY_EXPORTS = {
    "VisionArrayDataset": ("array", "VisionArrayDataset"),
    "VisionImageDataset": ("image", "VisionImageDataset"),
    "ImagePathDataset": ("image", "ImagePathDataset"),
    "default_eval_transforms": ("transforms", "default_eval_transforms"),
    "default_train_transforms": ("transforms", "default_train_transforms"),
    "to_tensor_normalized": ("transforms", "to_tensor_normalized"),
    "DataLoaderConfig": ("datamodule", "DataLoaderConfig"),
    "VisionDataModule": ("datamodule", "VisionDataModule"),
    "SyntheticAnomalyDataset": ("synthetic", "SyntheticAnomalyDataset"),
    "SyntheticItem": ("synthetic", "SyntheticItem"),
    "CorruptionsDataset": ("corruptions", "CorruptionsDataset"),
    "CorruptionItem": ("corruptions", "CorruptionItem"),
    # Dataset → manifest converters
    "convert_mvtec_ad2_to_manifest": ("mvtec_ad2", "convert_mvtec_ad2_to_manifest"),
    "convert_real_iad_to_manifest": ("real_iad", "convert_real_iad_to_manifest"),
    "convert_rad_to_manifest": ("rad", "convert_rad_to_manifest"),
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
