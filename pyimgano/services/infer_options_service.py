from __future__ import annotations

from typing import Any

from pyimgano.presets.catalog import resolve_defects_preset, resolve_preprocessing_preset


def apply_defects_defaults(args: Any, defects_payload: dict[str, Any]) -> None:
    """Apply defaults from an infer-config/run `defects` payload.

    Explicit CLI values win over payload defaults.
    """

    if not defects_payload:
        return

    def _coerce_roi(value: Any) -> list[float] | None:
        if value is None:
            return None
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            raise ValueError(
                "infer-config defects.roi_xyxy_norm must be a list of length 4 or null"
            )
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

    if args.roi_xyxy_norm is None:
        roi = _coerce_roi(defects_payload.get("roi_xyxy_norm", None))
        if roi is not None:
            args.roi_xyxy_norm = roi

    if int(getattr(args, "defect_min_area", 0)) == 0:
        v = _coerce_int(defects_payload.get("min_area", None), name="min_area")
        if v is not None:
            args.defect_min_area = int(v)

    if int(getattr(args, "defect_border_ignore_px", 0)) == 0:
        v = _coerce_int(defects_payload.get("border_ignore_px", None), name="border_ignore_px")
        if v is not None:
            args.defect_border_ignore_px = int(v)

    ms_raw = defects_payload.get("map_smoothing", None)
    if ms_raw is not None:
        if not isinstance(ms_raw, dict):
            raise ValueError("infer-config defects.map_smoothing must be a JSON object/dict.")
        if str(getattr(args, "defect_map_smoothing", "none")) == "none":
            v = _coerce_smoothing_method(ms_raw.get("method", None))
            if v is not None:
                args.defect_map_smoothing = str(v)
        if int(getattr(args, "defect_map_smoothing_ksize", 0)) == 0:
            v = _coerce_int(ms_raw.get("ksize", None), name="map_smoothing.ksize")
            if v is not None:
                args.defect_map_smoothing_ksize = int(v)
        if float(getattr(args, "defect_map_smoothing_sigma", 0.0)) == 0.0:
            v = _coerce_float(ms_raw.get("sigma", None), name="map_smoothing.sigma")
            if v is not None:
                args.defect_map_smoothing_sigma = float(v)

    hyst_raw = defects_payload.get("hysteresis", None)
    if hyst_raw is not None:
        if not isinstance(hyst_raw, dict):
            raise ValueError("infer-config defects.hysteresis must be a JSON object/dict.")

        enabled = hyst_raw.get("enabled", None)
        if enabled is True and not bool(getattr(args, "defect_hysteresis", False)):
            args.defect_hysteresis = True

        if getattr(args, "defect_hysteresis_low", None) is None:
            v = _coerce_float(hyst_raw.get("low", None), name="hysteresis.low")
            if v is not None:
                args.defect_hysteresis_low = float(v)

        if getattr(args, "defect_hysteresis_high", None) is None:
            v = _coerce_float(hyst_raw.get("high", None), name="hysteresis.high")
            if v is not None:
                args.defect_hysteresis_high = float(v)

    shape_raw = defects_payload.get("shape_filters", None)
    if shape_raw is not None:
        if not isinstance(shape_raw, dict):
            raise ValueError("infer-config defects.shape_filters must be a JSON object/dict.")

        if getattr(args, "defect_min_fill_ratio", None) is None:
            v = _coerce_float(
                shape_raw.get("min_fill_ratio", None), name="shape_filters.min_fill_ratio"
            )
            if v is not None:
                args.defect_min_fill_ratio = float(v)

        if getattr(args, "defect_max_aspect_ratio", None) is None:
            v = _coerce_float(
                shape_raw.get("max_aspect_ratio", None), name="shape_filters.max_aspect_ratio"
            )
            if v is not None:
                args.defect_max_aspect_ratio = float(v)

        if getattr(args, "defect_min_solidity", None) is None:
            v = _coerce_float(
                shape_raw.get("min_solidity", None), name="shape_filters.min_solidity"
            )
            if v is not None:
                args.defect_min_solidity = float(v)

    merge_raw = defects_payload.get("merge_nearby", None)
    if merge_raw is not None:
        if not isinstance(merge_raw, dict):
            raise ValueError("infer-config defects.merge_nearby must be a JSON object/dict.")

        enabled = merge_raw.get("enabled", None)
        if enabled is True and not bool(getattr(args, "defect_merge_nearby", False)):
            args.defect_merge_nearby = True

        if int(getattr(args, "defect_merge_nearby_max_gap_px", 0)) == 0:
            v = _coerce_int(merge_raw.get("max_gap_px", None), name="merge_nearby.max_gap_px")
            if v is not None:
                args.defect_merge_nearby_max_gap_px = int(v)

    if getattr(args, "defect_min_score_max", None) is None:
        v = _coerce_float(defects_payload.get("min_score_max", None), name="min_score_max")
        if v is not None:
            args.defect_min_score_max = float(v)

    if getattr(args, "defect_min_score_mean", None) is None:
        v = _coerce_float(defects_payload.get("min_score_mean", None), name="min_score_mean")
        if v is not None:
            args.defect_min_score_mean = float(v)

    if int(getattr(args, "defect_open_ksize", 0)) == 0:
        v = _coerce_int(defects_payload.get("open_ksize", None), name="open_ksize")
        if v is not None:
            args.defect_open_ksize = int(v)

    if int(getattr(args, "defect_close_ksize", 0)) == 0:
        v = _coerce_int(defects_payload.get("close_ksize", None), name="close_ksize")
        if v is not None:
            args.defect_close_ksize = int(v)

    if getattr(args, "defect_max_regions", None) is None:
        v = defects_payload.get("max_regions", None)
        if v is not None:
            args.defect_max_regions = int(v)

    if str(getattr(args, "defect_max_regions_sort_by", "score_max")) == "score_max":
        v = defects_payload.get("max_regions_sort_by", None)
        if v is not None:
            vv = str(v).lower().strip()
            if vv not in ("score_max", "score_mean", "area"):
                raise ValueError(
                    "infer-config defects.max_regions_sort_by must be one of: score_max|score_mean|area"
                )
            args.defect_max_regions_sort_by = vv

    if float(getattr(args, "pixel_normal_quantile", 0.999)) == 0.999:
        v = _coerce_float(
            defects_payload.get("pixel_normal_quantile", None), name="pixel_normal_quantile"
        )
        if v is not None:
            args.pixel_normal_quantile = float(v)

    if (
        str(getattr(args, "pixel_threshold_strategy", "normal_pixel_quantile"))
        == "normal_pixel_quantile"
    ):
        v = defects_payload.get("pixel_threshold_strategy", None)
        if v is not None:
            args.pixel_threshold_strategy = str(v)

    if str(getattr(args, "mask_format", "png")) == "png":
        v = _coerce_mask_format(defects_payload.get("mask_format", None))
        if v is not None:
            args.mask_format = str(v)

    if not bool(getattr(args, "defect_fill_holes", False)):
        v = _coerce_bool(defects_payload.get("fill_holes", None), name="fill_holes")
        if v is True:
            args.defect_fill_holes = True


def resolve_defects_preset_payload(name: str | None) -> dict[str, Any] | None:
    if name is None:
        return None

    preset = resolve_defects_preset(str(name))
    if preset is None:
        raise ValueError(f"Unknown defects preset: {name!r}")

    return dict(preset.payload)


def resolve_preprocessing_preset_knobs(name: str | None):
    if name is None:
        return None

    from pyimgano.inference.preprocessing import parse_illumination_contrast_knobs

    preset = resolve_preprocessing_preset(str(name))
    if preset is None:
        raise ValueError(f"Unknown preprocessing preset: {name!r}")
    if str(getattr(preset, "config_key", "")) != "preprocessing.illumination_contrast":
        raise ValueError(
            "Only preprocessing.illumination_contrast presets are currently supported by pyimgano-infer."
        )

    payload = dict(getattr(preset, "payload", {}) or {})
    return parse_illumination_contrast_knobs(payload)


__all__ = [
    "apply_defects_defaults",
    "resolve_defects_preset_payload",
    "resolve_preprocessing_preset_knobs",
]
