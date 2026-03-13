from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TilingConfig:
    """Configuration for high-resolution tiling inference (optional)."""

    tile_size: int | None = None
    stride: int | None = None
    score_reduce: str = "max"
    score_topk: float = 0.1
    map_reduce: str = "max"


@dataclass(frozen=True)
class MapPostprocessConfig:
    """Configuration for anomaly-map postprocessing (optional)."""

    normalize: bool = True
    normalize_method: str = "minmax"
    percentile_range: tuple[float, float] = (1.0, 99.0)
    gaussian_sigma: float = 0.0
    morph_open_ksize: int = 0
    morph_close_ksize: int = 0
    component_threshold: float | None = None
    min_component_area: int = 0


@dataclass(frozen=True)
class AdaptationConfig:
    tiling: TilingConfig = field(default_factory=TilingConfig)
    postprocess: MapPostprocessConfig | None = None
    save_maps: bool = False


__all__ = [
    "TilingConfig",
    "MapPostprocessConfig",
    "AdaptationConfig",
]
