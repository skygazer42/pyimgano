"""
Datasets module providing image data loading and preprocessing components.

This module includes dataset classes, data loaders, and transformation utilities
for vision-based anomaly detection tasks.
"""

from .array import VisionArrayDataset
from .benchmarks import (
    BTADDataset,
    BaseDataset,
    CustomDataset,
    DatasetInfo,
    MVTecAD2Dataset,
    MVTecDataset,
    MVTecLOCODataset,
    VisADataset,
    load_dataset,
)
from .datamodule import DataLoaderConfig, VisionDataModule
from .image import ImagePathDataset, VisionImageDataset
from .transforms import default_eval_transforms, default_train_transforms, to_tensor_normalized

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
