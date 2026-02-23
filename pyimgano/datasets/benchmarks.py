"""Benchmark dataset loaders.

This module exposes the standard anomaly-detection benchmark datasets used by
`pyimgano-benchmark` and `pyimgano.pipelines`.

These are currently implemented under `pyimgano.utils.datasets` for historical
reasons; this file provides a stable, torch-like surface under
`pyimgano.datasets`.
"""

from __future__ import annotations

from pyimgano.utils.datasets import (  # noqa: F401
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

__all__ = [
    "BaseDataset",
    "DatasetInfo",
    "MVTecDataset",
    "MVTecLOCODataset",
    "MVTecAD2Dataset",
    "VisADataset",
    "BTADDataset",
    "CustomDataset",
    "load_dataset",
]

