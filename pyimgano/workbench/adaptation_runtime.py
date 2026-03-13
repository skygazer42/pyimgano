from __future__ import annotations

from typing import Any

from pyimgano.workbench.adaptation_types import MapPostprocessConfig, TilingConfig


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
        component_threshold=(
            float(config.component_threshold) if config.component_threshold is not None else None
        ),
        min_component_area=int(config.min_component_area),
    )


__all__ = ["apply_tiling", "build_postprocess"]
