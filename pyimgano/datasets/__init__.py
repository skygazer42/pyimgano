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
    # Torch-style datasets / datamodule
    "ImagePathDataset",
    "VisionImageDataset",
    "VisionArrayDataset",
    "default_eval_transforms",
    "default_train_transforms",
    "to_tensor_normalized",
    "DataLoaderConfig",
    "VisionDataModule",
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
