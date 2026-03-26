from __future__ import annotations

from typing import Any, Mapping

from pyimgano.workbench.adaptation_types import AdaptationConfig, MapPostprocessConfig, TilingConfig
from pyimgano.workbench.config_parse_primitives import (
    _optional_float,
    _optional_int,
    _parse_percentile_range,
    _require_mapping,
)


def _parse_tiling_config(a_map: Mapping[str, Any]) -> TilingConfig:
    tiling_raw = a_map.get("tiling", None)
    if tiling_raw is None:
        return TilingConfig()

    t_map = _require_mapping(tiling_raw, name="adaptation.tiling")
    tile_size = _optional_int(t_map.get("tile_size", None), name="adaptation.tiling.tile_size")
    stride = _optional_int(t_map.get("stride", None), name="adaptation.tiling.stride")
    if tile_size is not None and tile_size <= 0:
        raise ValueError("adaptation.tiling.tile_size must be positive or null")
    if stride is not None and stride <= 0:
        raise ValueError("adaptation.tiling.stride must be positive or null")

    score_topk = _optional_float(
        t_map.get("score_topk", 0.1),
        name="adaptation.tiling.score_topk",
    )
    return TilingConfig(
        tile_size=tile_size,
        stride=stride,
        score_reduce=str(t_map.get("score_reduce", "max")),
        score_topk=float(score_topk if score_topk is not None else 0.1),
        map_reduce=str(t_map.get("map_reduce", "max")),
    )


def _parse_postprocess_config(a_map: Mapping[str, Any]) -> MapPostprocessConfig | None:
    post_raw = a_map.get("postprocess", None)
    if post_raw is None:
        return None

    p_map = _require_mapping(post_raw, name="adaptation.postprocess")
    component_threshold = _optional_float(
        p_map.get("component_threshold", None),
        name="adaptation.postprocess.component_threshold",
    )
    return MapPostprocessConfig(
        normalize=bool(p_map.get("normalize", True)),
        normalize_method=str(p_map.get("normalize_method", "minmax")),
        percentile_range=_parse_percentile_range(p_map.get("percentile_range", None)),
        gaussian_sigma=float(
            _optional_float(
                p_map.get("gaussian_sigma", 0.0),
                name="adaptation.postprocess.gaussian_sigma",
            )
            or 0.0
        ),
        morph_open_ksize=int(
            _optional_int(
                p_map.get("morph_open_ksize", 0),
                name="adaptation.postprocess.morph_open_ksize",
            )
            or 0
        ),
        morph_close_ksize=int(
            _optional_int(
                p_map.get("morph_close_ksize", 0),
                name="adaptation.postprocess.morph_close_ksize",
            )
            or 0
        ),
        component_threshold=component_threshold,
        min_component_area=int(
            _optional_int(
                p_map.get("min_component_area", 0),
                name="adaptation.postprocess.min_component_area",
            )
            or 0
        ),
    )


def _parse_adaptation_config(top: Mapping[str, Any]) -> AdaptationConfig:
    adaptation_raw = top.get("adaptation", None)
    if adaptation_raw is None:
        return AdaptationConfig()

    a_map = _require_mapping(adaptation_raw, name="adaptation")
    tiling = _parse_tiling_config(a_map)
    postprocess = _parse_postprocess_config(a_map)

    return AdaptationConfig(
        tiling=tiling,
        postprocess=postprocess,
        save_maps=bool(a_map.get("save_maps", False)),
    )


__all__ = ["_parse_adaptation_config"]
