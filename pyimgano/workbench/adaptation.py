from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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


def apply_tiling(detector: Any, tiling: TilingConfig) -> Any:
    if tiling.tile_size is None:
        return detector

    from pyimgano.inference.tiling import TiledDetector

    return TiledDetector(
        detector=detector,
        tile_size=int(tiling.tile_size),
        stride=(int(tiling.stride) if tiling.stride is not None else None),
        score_reduce=str(tiling.score_reduce),
        score_topk=float(tiling.score_topk),
        map_reduce=str(tiling.map_reduce),
    )


def build_postprocess(config: MapPostprocessConfig | None):
    if config is None:
        return None

    from pyimgano.postprocess.anomaly_map import AnomalyMapPostprocess

    return AnomalyMapPostprocess(
        normalize=bool(config.normalize),
        normalize_method=str(config.normalize_method),
        percentile_range=(float(config.percentile_range[0]), float(config.percentile_range[1])),
        gaussian_sigma=float(config.gaussian_sigma),
        morph_open_ksize=int(config.morph_open_ksize),
        morph_close_ksize=int(config.morph_close_ksize),
        component_threshold=(float(config.component_threshold) if config.component_threshold is not None else None),
        min_component_area=int(config.min_component_area),
    )
