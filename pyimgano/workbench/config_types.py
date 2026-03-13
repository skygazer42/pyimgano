from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pyimgano.preprocessing.industrial_presets import IlluminationContrastKnobs
from pyimgano.workbench.adaptation_types import AdaptationConfig


@dataclass(frozen=True)
class SplitPolicyConfig:
    """Controls auto-splitting for datasets that support it (e.g. manifest JSONL)."""

    mode: str = "benchmark"
    scope: str = "category"
    seed: int | None = None
    test_normal_fraction: float = 0.2


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    root: str
    manifest_path: str | None = None
    category: str = "all"
    resize: tuple[int, int] = (256, 256)
    input_mode: str = "paths"
    limit_train: int | None = None
    limit_test: int | None = None
    split_policy: SplitPolicyConfig = field(default_factory=SplitPolicyConfig)


@dataclass(frozen=True)
class ModelConfig:
    name: str
    device: str = "cpu"
    preset: str | None = None
    pretrained: bool = True
    contamination: float = 0.1
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    checkpoint_path: str | None = None


@dataclass(frozen=True)
class OutputConfig:
    output_dir: str | None = None
    save_run: bool = True
    per_image_jsonl: bool = True


@dataclass(frozen=True)
class TrainingConfig:
    enabled: bool = False
    epochs: int | None = None
    lr: float | None = None
    checkpoint_name: str = "model.pt"


@dataclass(frozen=True)
class MapSmoothingConfig:
    method: str = "none"
    ksize: int = 0
    sigma: float = 0.0


@dataclass(frozen=True)
class HysteresisConfig:
    enabled: bool = False
    low: float | None = None
    high: float | None = None


@dataclass(frozen=True)
class ShapeFiltersConfig:
    min_fill_ratio: float | None = None
    max_aspect_ratio: float | None = None
    min_solidity: float | None = None


@dataclass(frozen=True)
class MergeNearbyConfig:
    enabled: bool = False
    max_gap_px: int = 0


@dataclass(frozen=True)
class DefectsConfig:
    enabled: bool = False
    pixel_threshold: float | None = None
    pixel_threshold_strategy: str = "normal_pixel_quantile"
    pixel_normal_quantile: float = 0.999
    mask_format: str = "png"
    roi_xyxy_norm: tuple[float, float, float, float] | None = None
    border_ignore_px: int = 0
    map_smoothing: MapSmoothingConfig = field(default_factory=MapSmoothingConfig)
    hysteresis: HysteresisConfig = field(default_factory=HysteresisConfig)
    shape_filters: ShapeFiltersConfig = field(default_factory=ShapeFiltersConfig)
    merge_nearby: MergeNearbyConfig = field(default_factory=MergeNearbyConfig)
    min_area: int = 0
    min_score_max: float | None = None
    min_score_mean: float | None = None
    open_ksize: int = 0
    close_ksize: int = 0
    fill_holes: bool = False
    max_regions: int | None = None
    max_regions_sort_by: str = "score_max"


@dataclass(frozen=True)
class PreprocessingConfig:
    """Optional preprocessing configuration for industrial runs."""

    illumination_contrast: IlluminationContrastKnobs | None = None


@dataclass(frozen=True)
class WorkbenchConfig:
    dataset: DatasetConfig
    model: ModelConfig
    recipe: str = "industrial-adapt"
    seed: int | None = None
    output: OutputConfig = field(default_factory=OutputConfig)
    adaptation: AdaptationConfig = field(default_factory=AdaptationConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    defects: DefectsConfig = field(default_factory=DefectsConfig)


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
    "PreprocessingConfig",
    "WorkbenchConfig",
]
