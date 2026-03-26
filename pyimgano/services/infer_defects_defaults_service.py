from __future__ import annotations

from typing import Any


def _coerce_roi(value: Any) -> list[float] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("infer-config defects.roi_xyxy_norm must be a list of length 4 or null")
    try:
        return [float(v) for v in value]
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        raise ValueError(
            f"infer-config defects.roi_xyxy_norm must contain floats, got {value!r}"
        ) from exc


def _coerce_bool(value: Any, *, name: str) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return bool(value)
    raise ValueError(f"infer-config defects.{name} must be a boolean, got {value!r}")


def _coerce_int(value: Any, *, name: str) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        raise ValueError(f"infer-config defects.{name} must be an int, got {value!r}") from exc


def _coerce_float(value: Any, *, name: str) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        raise ValueError(f"infer-config defects.{name} must be a float, got {value!r}") from exc


def _coerce_mask_format(value: Any) -> str | None:
    if value is None:
        return None
    fmt = str(value)
    if fmt not in ("png", "npy"):
        raise ValueError("infer-config defects.mask_format must be 'png' or 'npy'")
    return fmt


def _coerce_smoothing_method(value: Any) -> str | None:
    if value is None:
        return None
    method = str(value).lower().strip()
    if method in ("none", "median", "gaussian", "box"):
        return method
    raise ValueError(
        "infer-config defects.map_smoothing.method must be one of: none|median|gaussian|box"
    )


def _coerce_max_regions_sort_by(value: Any) -> str | None:
    if value is None:
        return None
    sort_by = str(value).lower().strip()
    if sort_by not in ("score_max", "score_mean", "area"):
        raise ValueError(
            "infer-config defects.max_regions_sort_by must be one of: score_max|score_mean|area"
        )
    return sort_by


def _require_nested_payload(value: Any, *, name: str) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"infer-config defects.{name} must be a JSON object/dict.")
    return value


def _apply_roi_defaults(args: Any, defects_payload: dict[str, Any]) -> None:
    if args.roi_xyxy_norm is not None:
        return
    roi = _coerce_roi(defects_payload.get("roi_xyxy_norm"))
    if roi is not None:
        args.roi_xyxy_norm = roi


def _apply_area_and_kernel_defaults(args: Any, defects_payload: dict[str, Any]) -> None:
    if int(getattr(args, "defect_min_area", 0)) == 0:
        min_area = _coerce_int(defects_payload.get("min_area"), name="min_area")
        if min_area is not None:
            args.defect_min_area = int(min_area)

    if int(getattr(args, "defect_border_ignore_px", 0)) == 0:
        border_ignore_px = _coerce_int(
            defects_payload.get("border_ignore_px"), name="border_ignore_px"
        )
        if border_ignore_px is not None:
            args.defect_border_ignore_px = int(border_ignore_px)

    if int(getattr(args, "defect_open_ksize", 0)) == 0:
        open_ksize = _coerce_int(defects_payload.get("open_ksize"), name="open_ksize")
        if open_ksize is not None:
            args.defect_open_ksize = int(open_ksize)

    if int(getattr(args, "defect_close_ksize", 0)) == 0:
        close_ksize = _coerce_int(defects_payload.get("close_ksize"), name="close_ksize")
        if close_ksize is not None:
            args.defect_close_ksize = int(close_ksize)


def _apply_map_smoothing_defaults(args: Any, defects_payload: dict[str, Any]) -> None:
    map_smoothing = _require_nested_payload(
        defects_payload.get("map_smoothing"), name="map_smoothing"
    )
    if map_smoothing is None:
        return

    if str(getattr(args, "defect_map_smoothing", "none")) == "none":
        method = _coerce_smoothing_method(
            map_smoothing.get("method"),
        )
        if method is not None:
            args.defect_map_smoothing = str(method)

    if int(getattr(args, "defect_map_smoothing_ksize", 0)) == 0:
        ksize = _coerce_int(map_smoothing.get("ksize"), name="map_smoothing.ksize")
        if ksize is not None:
            args.defect_map_smoothing_ksize = int(ksize)

    if getattr(args, "defect_map_smoothing_sigma", None) is None:
        sigma = _coerce_float(map_smoothing.get("sigma"), name="map_smoothing.sigma")
        if sigma is not None:
            args.defect_map_smoothing_sigma = float(sigma)


def _apply_hysteresis_defaults(args: Any, defects_payload: dict[str, Any]) -> None:
    hysteresis = _require_nested_payload(defects_payload.get("hysteresis"), name="hysteresis")
    if hysteresis is None:
        return

    enabled = hysteresis.get("enabled")
    if enabled is True and not bool(getattr(args, "defect_hysteresis", False)):
        args.defect_hysteresis = True

    if getattr(args, "defect_hysteresis_low", None) is None:
        low = _coerce_float(hysteresis.get("low"), name="hysteresis.low")
        if low is not None:
            args.defect_hysteresis_low = float(low)

    if getattr(args, "defect_hysteresis_high", None) is None:
        high = _coerce_float(hysteresis.get("high"), name="hysteresis.high")
        if high is not None:
            args.defect_hysteresis_high = float(high)


def _apply_shape_filter_defaults(args: Any, defects_payload: dict[str, Any]) -> None:
    shape_filters = _require_nested_payload(
        defects_payload.get("shape_filters"), name="shape_filters"
    )
    if shape_filters is None:
        return

    if getattr(args, "defect_min_fill_ratio", None) is None:
        min_fill_ratio = _coerce_float(
            shape_filters.get("min_fill_ratio"), name="shape_filters.min_fill_ratio"
        )
        if min_fill_ratio is not None:
            args.defect_min_fill_ratio = float(min_fill_ratio)

    if getattr(args, "defect_max_aspect_ratio", None) is None:
        max_aspect_ratio = _coerce_float(
            shape_filters.get("max_aspect_ratio"), name="shape_filters.max_aspect_ratio"
        )
        if max_aspect_ratio is not None:
            args.defect_max_aspect_ratio = float(max_aspect_ratio)

    if getattr(args, "defect_min_solidity", None) is None:
        min_solidity = _coerce_float(
            shape_filters.get("min_solidity"), name="shape_filters.min_solidity"
        )
        if min_solidity is not None:
            args.defect_min_solidity = float(min_solidity)


def _apply_merge_nearby_defaults(args: Any, defects_payload: dict[str, Any]) -> None:
    merge_nearby = _require_nested_payload(defects_payload.get("merge_nearby"), name="merge_nearby")
    if merge_nearby is None:
        return

    enabled = merge_nearby.get("enabled")
    if enabled is True and not bool(getattr(args, "defect_merge_nearby", False)):
        args.defect_merge_nearby = True

    if int(getattr(args, "defect_merge_nearby_max_gap_px", 0)) == 0:
        max_gap_px = _coerce_int(merge_nearby.get("max_gap_px"), name="merge_nearby.max_gap_px")
        if max_gap_px is not None:
            args.defect_merge_nearby_max_gap_px = int(max_gap_px)


def _apply_score_and_limit_defaults(args: Any, defects_payload: dict[str, Any]) -> None:
    if getattr(args, "defect_min_score_max", None) is None:
        min_score_max = _coerce_float(defects_payload.get("min_score_max"), name="min_score_max")
        if min_score_max is not None:
            args.defect_min_score_max = float(min_score_max)

    if getattr(args, "defect_min_score_mean", None) is None:
        min_score_mean = _coerce_float(defects_payload.get("min_score_mean"), name="min_score_mean")
        if min_score_mean is not None:
            args.defect_min_score_mean = float(min_score_mean)

    if getattr(args, "defect_max_regions", None) is None:
        max_regions = defects_payload.get("max_regions")
        if max_regions is not None:
            args.defect_max_regions = int(max_regions)

    if str(getattr(args, "defect_max_regions_sort_by", "score_max")) == "score_max":
        max_regions_sort_by = _coerce_max_regions_sort_by(
            defects_payload.get("max_regions_sort_by")
        )
        if max_regions_sort_by is not None:
            args.defect_max_regions_sort_by = max_regions_sort_by


def _apply_mask_and_threshold_strategy_defaults(args: Any, defects_payload: dict[str, Any]) -> None:
    if getattr(args, "pixel_normal_quantile", None) is None:
        pixel_normal_quantile = _coerce_float(
            defects_payload.get("pixel_normal_quantile"), name="pixel_normal_quantile"
        )
        if pixel_normal_quantile is not None:
            args.pixel_normal_quantile = float(pixel_normal_quantile)

    if (
        str(getattr(args, "pixel_threshold_strategy", "normal_pixel_quantile"))
        == "normal_pixel_quantile"
    ):
        pixel_threshold_strategy = defects_payload.get("pixel_threshold_strategy")
        if pixel_threshold_strategy is not None:
            args.pixel_threshold_strategy = str(pixel_threshold_strategy)

    if str(getattr(args, "mask_format", "png")) == "png":
        mask_format = _coerce_mask_format(defects_payload.get("mask_format"))
        if mask_format is not None:
            args.mask_format = str(mask_format)

    if not bool(getattr(args, "defect_fill_holes", False)):
        fill_holes = _coerce_bool(defects_payload.get("fill_holes"), name="fill_holes")
        if fill_holes is True:
            args.defect_fill_holes = True


_DEFECT_DEFAULT_APPLIERS = (
    _apply_roi_defaults,
    _apply_area_and_kernel_defaults,
    _apply_map_smoothing_defaults,
    _apply_hysteresis_defaults,
    _apply_shape_filter_defaults,
    _apply_merge_nearby_defaults,
    _apply_score_and_limit_defaults,
    _apply_mask_and_threshold_strategy_defaults,
)


def apply_defects_defaults(args: Any, defects_payload: dict[str, Any]) -> None:
    """Apply defaults from an infer-config/run `defects` payload.

    Explicit CLI values win over payload defaults.
    """

    if not defects_payload:
        return

    for applier in _DEFECT_DEFAULT_APPLIERS:
        applier(args, defects_payload)


__all__ = ["apply_defects_defaults"]
