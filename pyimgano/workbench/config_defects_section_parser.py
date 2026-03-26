from __future__ import annotations

from typing import Any, Mapping

from pyimgano.workbench.config_parse_primitives import (
    _optional_float,
    _optional_int,
    _parse_roi_xyxy_norm,
    _require_mapping,
)
from pyimgano.workbench.config_types import (
    DefectsConfig,
    HysteresisConfig,
    MapSmoothingConfig,
    MergeNearbyConfig,
    ShapeFiltersConfig,
)


def _parse_map_smoothing_config(d_map: Mapping[str, Any]) -> MapSmoothingConfig:
    ms_raw = d_map.get("map_smoothing", None)
    if ms_raw is None:
        return MapSmoothingConfig()

    ms_map = _require_mapping(ms_raw, name="defects.map_smoothing")
    ms_method = str(ms_map.get("method", "none")).lower().strip()
    if ms_method not in ("none", "median", "gaussian", "box"):
        raise ValueError("defects.map_smoothing.method must be one of: none|median|gaussian|box")
    ms_ksize = int(_optional_int(ms_map.get("ksize", 0), name="defects.map_smoothing.ksize") or 0)
    ms_sigma = _optional_float(ms_map.get("sigma", 0.0), name="defects.map_smoothing.sigma")
    ms_sigma_v = float(ms_sigma if ms_sigma is not None else 0.0)
    if ms_ksize < 0:
        raise ValueError("defects.map_smoothing.ksize must be >= 0")
    if ms_sigma_v < 0.0:
        raise ValueError("defects.map_smoothing.sigma must be >= 0")

    if ms_method in ("median", "box") and ms_ksize not in (0, 1) and ms_ksize < 3:
        raise ValueError("defects.map_smoothing.ksize must be >= 3 for median/box smoothing")

    return MapSmoothingConfig(
        method=ms_method,
        ksize=ms_ksize,
        sigma=ms_sigma_v,
    )


def _parse_hysteresis_config(d_map: Mapping[str, Any]) -> HysteresisConfig:
    hyst_raw = d_map.get("hysteresis", None)
    if hyst_raw is None:
        return HysteresisConfig()

    hyst_map = _require_mapping(hyst_raw, name="defects.hysteresis")
    hyst_enabled = bool(hyst_map.get("enabled", False))
    hyst_low = _optional_float(hyst_map.get("low", None), name="defects.hysteresis.low")
    hyst_high = _optional_float(hyst_map.get("high", None), name="defects.hysteresis.high")
    if hyst_low is not None and float(hyst_low) < 0.0:
        raise ValueError("defects.hysteresis.low must be >= 0 or null")
    if hyst_high is not None and float(hyst_high) < 0.0:
        raise ValueError("defects.hysteresis.high must be >= 0 or null")
    return HysteresisConfig(
        enabled=hyst_enabled,
        low=(float(hyst_low) if hyst_low is not None else None),
        high=(float(hyst_high) if hyst_high is not None else None),
    )


def _parse_shape_filters_config(d_map: Mapping[str, Any]) -> ShapeFiltersConfig:
    sf_raw = d_map.get("shape_filters", None)
    if sf_raw is None:
        return ShapeFiltersConfig()

    sf_map = _require_mapping(sf_raw, name="defects.shape_filters")
    sf_min_fill_ratio = _optional_float(
        sf_map.get("min_fill_ratio", None),
        name="defects.shape_filters.min_fill_ratio",
    )
    sf_max_aspect_ratio = _optional_float(
        sf_map.get("max_aspect_ratio", None),
        name="defects.shape_filters.max_aspect_ratio",
    )
    sf_min_solidity = _optional_float(
        sf_map.get("min_solidity", None),
        name="defects.shape_filters.min_solidity",
    )

    if sf_min_fill_ratio is not None and not (0.0 <= float(sf_min_fill_ratio) <= 1.0):
        raise ValueError("defects.shape_filters.min_fill_ratio must be in [0,1] or null")
    if sf_max_aspect_ratio is not None and float(sf_max_aspect_ratio) < 1.0:
        raise ValueError("defects.shape_filters.max_aspect_ratio must be >= 1.0 or null")
    if sf_min_solidity is not None and not (0.0 <= float(sf_min_solidity) <= 1.0):
        raise ValueError("defects.shape_filters.min_solidity must be in [0,1] or null")

    return ShapeFiltersConfig(
        min_fill_ratio=(float(sf_min_fill_ratio) if sf_min_fill_ratio is not None else None),
        max_aspect_ratio=(float(sf_max_aspect_ratio) if sf_max_aspect_ratio is not None else None),
        min_solidity=(float(sf_min_solidity) if sf_min_solidity is not None else None),
    )


def _parse_merge_nearby_config(d_map: Mapping[str, Any]) -> MergeNearbyConfig:
    merge_raw = d_map.get("merge_nearby", None)
    if merge_raw is None:
        return MergeNearbyConfig()

    merge_map = _require_mapping(merge_raw, name="defects.merge_nearby")
    merge_enabled = bool(merge_map.get("enabled", False))
    merge_gap = int(
        _optional_int(
            merge_map.get("max_gap_px", 0),
            name="defects.merge_nearby.max_gap_px",
        )
        or 0
    )
    if merge_gap < 0:
        raise ValueError("defects.merge_nearby.max_gap_px must be >= 0")
    return MergeNearbyConfig(enabled=merge_enabled, max_gap_px=merge_gap)


def _validate_defects_config_values(
    *,
    border_ignore_px: int,
    min_area: int,
    open_ksize: int,
    close_ksize: int,
    max_regions: int | None,
    max_regions_sort_by: str,
    min_score_max: float | None,
    min_score_mean: float | None,
    pixel_normal_quantile: float,
    mask_format: str,
) -> None:
    if border_ignore_px < 0:
        raise ValueError("defects.border_ignore_px must be >= 0")
    if min_area < 0:
        raise ValueError("defects.min_area must be >= 0")
    if open_ksize < 0:
        raise ValueError("defects.open_ksize must be >= 0")
    if close_ksize < 0:
        raise ValueError("defects.close_ksize must be >= 0")
    if max_regions is not None and max_regions <= 0:
        raise ValueError("defects.max_regions must be positive or null")
    if max_regions_sort_by not in ("score_max", "score_mean", "area"):
        raise ValueError("defects.max_regions_sort_by must be one of: score_max|score_mean|area")
    if min_score_max is not None and float(min_score_max) < 0.0:
        raise ValueError("defects.min_score_max must be >= 0 or null")
    if min_score_mean is not None and float(min_score_mean) < 0.0:
        raise ValueError("defects.min_score_mean must be >= 0 or null")
    if not (0.0 < float(pixel_normal_quantile) <= 1.0):
        raise ValueError("defects.pixel_normal_quantile must be in (0,1]")
    if mask_format not in ("png", "npy"):
        raise ValueError("defects.mask_format must be 'png' or 'npy'")


def _parse_defects_config(top: Mapping[str, Any]) -> DefectsConfig:
    defects_raw = top.get("defects", None)
    if defects_raw is None:
        return DefectsConfig()

    d_map = _require_mapping(defects_raw, name="defects")
    pixel_threshold = _optional_float(
        d_map.get("pixel_threshold", None),
        name="defects.pixel_threshold",
    )
    max_regions = _optional_int(d_map.get("max_regions", None), name="defects.max_regions")
    max_regions_sort_by = str(d_map.get("max_regions_sort_by", "score_max")).lower().strip()
    border_ignore_px = int(
        _optional_int(d_map.get("border_ignore_px", 0), name="defects.border_ignore_px") or 0
    )
    min_area = int(_optional_int(d_map.get("min_area", 0), name="defects.min_area") or 0)
    min_score_max = _optional_float(
        d_map.get("min_score_max", None),
        name="defects.min_score_max",
    )
    min_score_mean = _optional_float(
        d_map.get("min_score_mean", None),
        name="defects.min_score_mean",
    )
    open_ksize = int(_optional_int(d_map.get("open_ksize", 0), name="defects.open_ksize") or 0)
    close_ksize = int(_optional_int(d_map.get("close_ksize", 0), name="defects.close_ksize") or 0)

    q = _optional_float(
        d_map.get("pixel_normal_quantile", 0.999),
        name="defects.pixel_normal_quantile",
    )
    qv = float(q if q is not None else 0.999)
    mask_format = str(d_map.get("mask_format", "png"))
    _validate_defects_config_values(
        border_ignore_px=border_ignore_px,
        min_area=min_area,
        open_ksize=open_ksize,
        close_ksize=close_ksize,
        max_regions=max_regions,
        max_regions_sort_by=max_regions_sort_by,
        min_score_max=(float(min_score_max) if min_score_max is not None else None),
        min_score_mean=(float(min_score_mean) if min_score_mean is not None else None),
        pixel_normal_quantile=qv,
        mask_format=mask_format,
    )

    return DefectsConfig(
        enabled=bool(d_map.get("enabled", False)),
        pixel_threshold=(float(pixel_threshold) if pixel_threshold is not None else None),
        pixel_threshold_strategy=str(
            d_map.get("pixel_threshold_strategy", "normal_pixel_quantile")
        ),
        pixel_normal_quantile=qv,
        mask_format=mask_format,
        roi_xyxy_norm=_parse_roi_xyxy_norm(d_map.get("roi_xyxy_norm", None)),
        border_ignore_px=border_ignore_px,
        map_smoothing=_parse_map_smoothing_config(d_map),
        hysteresis=_parse_hysteresis_config(d_map),
        shape_filters=_parse_shape_filters_config(d_map),
        merge_nearby=_parse_merge_nearby_config(d_map),
        min_area=min_area,
        min_score_max=(float(min_score_max) if min_score_max is not None else None),
        min_score_mean=(float(min_score_mean) if min_score_mean is not None else None),
        open_ksize=open_ksize,
        close_ksize=close_ksize,
        fill_holes=bool(d_map.get("fill_holes", False)),
        max_regions=max_regions,
        max_regions_sort_by=max_regions_sort_by,
    )


__all__ = ["_parse_defects_config"]
