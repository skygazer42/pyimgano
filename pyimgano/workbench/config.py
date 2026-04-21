"""Compatibility facade for workbench config types and parsing."""

from __future__ import annotations

from typing import Any, Mapping, TypeVar

from pyimgano.workbench.config_parser import build_workbench_config_from_dict
from pyimgano.workbench.config_types import (
    DatasetConfig,
    DefectsConfig,
    HysteresisConfig,
    MapSmoothingConfig,
    MergeNearbyConfig,
    MetaConfig,
    ModelConfig,
    OutputConfig,
    PredictionConfig,
    PreprocessingConfig,
    ShapeFiltersConfig,
    SplitPolicyConfig,
    TrainingConfig,
    WorkbenchConfig,
)

WorkbenchConfigT = TypeVar("WorkbenchConfigT", bound=WorkbenchConfig)


def _workbench_config_from_dict(
    cls: type[WorkbenchConfigT],
    raw: Mapping[str, Any],
) -> WorkbenchConfigT:
    return build_workbench_config_from_dict(raw, config_cls=cls)


WorkbenchConfig.from_dict = classmethod(_workbench_config_from_dict)  # type: ignore[attr-defined]


__all__ = [
    "SplitPolicyConfig",
    "DatasetConfig",
    "ModelConfig",
    "OutputConfig",
    "TrainingConfig",
    "MapSmoothingConfig",
    "HysteresisConfig",
    "ShapeFiltersConfig",
    "MergeNearbyConfig",
    "DefectsConfig",
    "PredictionConfig",
    "MetaConfig",
    "PreprocessingConfig",
    "WorkbenchConfig",
]
