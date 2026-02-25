from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from pyimgano.inference.config import (
    load_infer_config,
    resolve_infer_checkpoint_path,
    select_infer_category,
)


_ALLOWED_PIXEL_THRESHOLD_STRATEGIES = {"normal_pixel_quantile", "fixed", "infer_config"}
_ALLOWED_MASK_FORMATS = {"png", "npy", "npz"}
_ALLOWED_MAP_SMOOTHING_METHODS = {"none", "median", "gaussian", "box"}
_ALLOWED_TILING_SCORE_REDUCE = {"max", "mean", "topk_mean"}
_ALLOWED_TILING_MAP_REDUCE = {"max", "mean", "hann", "gaussian"}
_ALLOWED_WHITE_BALANCE = {"none", "gray_world", "max_rgb"}


@dataclass(frozen=True)
class InferConfigValidation:
    payload: dict[str, Any]
    resolved_checkpoint_path: Path | None
    warnings: list[str]


def _require_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"infer-config key {name!r} must be a JSON object/dict, got {type(value).__name__}")
    return value


def _optional_mapping(value: Any, *, name: str) -> Mapping[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"infer-config key {name!r} must be a JSON object/dict, got {type(value).__name__}")
    return value


def _coerce_bool(value: Any, *, name: str) -> bool:
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(int(value))
    raise ValueError(f"infer-config key {name!r} must be a boolean, got {value!r}")


def _coerce_int(value: Any, *, name: str) -> int:
    try:
        return int(value)
    except Exception as exc:  # noqa: BLE001 - validation boundary
        raise ValueError(f"infer-config key {name!r} must be an int, got {value!r}") from exc


def _coerce_float(value: Any, *, name: str) -> float:
    try:
        return float(value)
    except Exception as exc:  # noqa: BLE001 - validation boundary
        raise ValueError(f"infer-config key {name!r} must be a float, got {value!r}") from exc


def _coerce_str(value: Any, *, name: str) -> str:
    if value is None:
        raise ValueError(f"infer-config key {name!r} must be a string, got null")
    return str(value)


def _coerce_optional_float(value: Any, *, name: str) -> float | None:
    if value is None:
        return None
    return _coerce_float(value, name=name)


def _coerce_optional_int(value: Any, *, name: str) -> int | None:
    if value is None:
        return None
    return _coerce_int(value, name=name)


def validate_infer_config_payload(
    payload: Mapping[str, Any],
    *,
    config_path: str | Path | None = None,
    category: str | None = None,
    check_files: bool = True,
) -> InferConfigValidation:
    """Validate and normalize an infer-config payload.

    This function is intentionally permissive: it validates known keys and
    best-effort coerces values into the types expected by `pyimgano-infer`.

    Args:
        payload: Parsed JSON object (dict-like).
        config_path: Path to the infer-config file (used for checkpoint resolution).
        category: Optional category selector for multi-category infer-configs.
        check_files: If True, validate referenced checkpoints exist on disk.

    Returns:
        InferConfigValidation containing the normalized payload, resolved checkpoint
        path (if present), and a list of warnings.
    """

    if not isinstance(payload, Mapping):
        raise ValueError(
            f"infer-config payload must be a JSON object/dict, got {type(payload).__name__}"
        )

    warnings: list[str] = []
    normalized: dict[str, Any] = dict(payload)

    normalized = select_infer_category(normalized, category=category)

    # Schema/version fields are additive; validate if present.
    if "schema_version" in normalized and normalized["schema_version"] is not None:
        normalized["schema_version"] = _coerce_int(
            normalized["schema_version"], name="schema_version"
        )

    if "pyimgano_version" in normalized and normalized["pyimgano_version"] is not None:
        normalized["pyimgano_version"] = str(normalized["pyimgano_version"])

    model_raw = normalized.get("model", None)
    model_map = _require_mapping(model_raw, name="model")
    model: dict[str, Any] = dict(model_map)

    model_name = model.get("name", None)
    if model_name is None:
        raise ValueError("infer-config model.name is required.")
    model["name"] = _coerce_str(model_name, name="model.name").strip()
    if not model["name"]:
        raise ValueError("infer-config model.name must be non-empty.")

    if "device" in model and model["device"] is not None:
        model["device"] = str(model["device"])
    if "preset" in model and model["preset"] is not None:
        model["preset"] = str(model["preset"])

    if "contamination" in model and model["contamination"] is not None:
        model["contamination"] = _coerce_float(model["contamination"], name="model.contamination")
    if "pretrained" in model and model["pretrained"] is not None:
        model["pretrained"] = _coerce_bool(model["pretrained"], name="model.pretrained")

    model_kwargs_raw = model.get("model_kwargs", None)
    if model_kwargs_raw is None:
        model["model_kwargs"] = {}
    else:
        model_kwargs_map = _require_mapping(model_kwargs_raw, name="model.model_kwargs")
        model["model_kwargs"] = dict(model_kwargs_map)

    if "checkpoint_path" in model and model["checkpoint_path"] is not None:
        model["checkpoint_path"] = str(model["checkpoint_path"])

    normalized["model"] = model

    preprocessing_raw = normalized.get("preprocessing", None)
    preprocessing_map = _optional_mapping(preprocessing_raw, name="preprocessing")
    if preprocessing_map is not None:
        preprocessing: dict[str, Any] = dict(preprocessing_map)

        ic_raw = preprocessing.get("illumination_contrast", None)
        ic_map = _optional_mapping(ic_raw, name="preprocessing.illumination_contrast")
        if ic_map is not None:
            ic: dict[str, Any] = dict(ic_map)

            if "white_balance" in ic and ic["white_balance"] is not None:
                wb = str(ic["white_balance"]).strip().lower()
                if wb in ("", "none"):
                    wb = "none"
                elif wb in ("gray_world", "gray-world", "grayworld"):
                    wb = "gray_world"
                elif wb in ("max_rgb", "max-rgb", "maxrgb"):
                    wb = "max_rgb"
                if wb not in _ALLOWED_WHITE_BALANCE:
                    raise ValueError(
                        "infer-config preprocessing.illumination_contrast.white_balance must be one of: "
                        f"{sorted(_ALLOWED_WHITE_BALANCE)}"
                    )
                ic["white_balance"] = wb

            for key in (
                "homomorphic",
                "homomorphic_per_channel",
                "clahe",
                "contrast_stretch",
            ):
                if key in ic and ic[key] is not None:
                    ic[key] = _coerce_bool(ic[key], name=f"preprocessing.illumination_contrast.{key}")

            if "homomorphic_cutoff" in ic and ic["homomorphic_cutoff"] is not None:
                cutoff = _coerce_float(
                    ic["homomorphic_cutoff"],
                    name="preprocessing.illumination_contrast.homomorphic_cutoff",
                )
                if not (0.0 < float(cutoff) <= 1.0):
                    raise ValueError(
                        "infer-config preprocessing.illumination_contrast.homomorphic_cutoff must be in (0,1]."
                    )
                ic["homomorphic_cutoff"] = float(cutoff)

            for key in (
                "homomorphic_gamma_low",
                "homomorphic_gamma_high",
                "homomorphic_c",
                "clahe_clip_limit",
                "contrast_lower_percentile",
                "contrast_upper_percentile",
            ):
                if key in ic and ic[key] is not None:
                    ic[key] = _coerce_float(ic[key], name=f"preprocessing.illumination_contrast.{key}")

            if "gamma" in ic and ic["gamma"] is not None:
                gamma = _coerce_float(ic["gamma"], name="preprocessing.illumination_contrast.gamma")
                if float(gamma) <= 0.0:
                    raise ValueError(
                        "infer-config preprocessing.illumination_contrast.gamma must be > 0."
                    )
                ic["gamma"] = float(gamma)

            if "clahe_tile_grid_size" in ic and ic["clahe_tile_grid_size"] is not None:
                tgs = ic["clahe_tile_grid_size"]
                if not isinstance(tgs, (list, tuple)) or len(tgs) != 2:
                    raise ValueError(
                        "infer-config preprocessing.illumination_contrast.clahe_tile_grid_size must be a list of length 2."
                    )
                a = _coerce_int(
                    tgs[0],
                    name="preprocessing.illumination_contrast.clahe_tile_grid_size[0]",
                )
                b = _coerce_int(
                    tgs[1],
                    name="preprocessing.illumination_contrast.clahe_tile_grid_size[1]",
                )
                if a <= 0 or b <= 0:
                    raise ValueError(
                        "infer-config preprocessing.illumination_contrast.clahe_tile_grid_size must be positive."
                    )
                ic["clahe_tile_grid_size"] = [int(a), int(b)]

            # Best-effort range validation (optional keys).
            if "clahe_clip_limit" in ic and ic["clahe_clip_limit"] is not None:
                if float(ic["clahe_clip_limit"]) <= 0.0:
                    raise ValueError(
                        "infer-config preprocessing.illumination_contrast.clahe_clip_limit must be > 0."
                    )

            if (
                "contrast_lower_percentile" in ic
                and "contrast_upper_percentile" in ic
                and ic.get("contrast_lower_percentile", None) is not None
                and ic.get("contrast_upper_percentile", None) is not None
            ):
                lo = float(ic["contrast_lower_percentile"])
                hi = float(ic["contrast_upper_percentile"])
                if not (0.0 <= lo <= 100.0 and 0.0 <= hi <= 100.0 and lo < hi):
                    raise ValueError(
                        "infer-config preprocessing.illumination_contrast contrast percentiles must satisfy 0<=lower<upper<=100."
                    )

            preprocessing["illumination_contrast"] = ic

        normalized["preprocessing"] = preprocessing

    adaptation_raw = normalized.get("adaptation", None)
    adaptation_map = _optional_mapping(adaptation_raw, name="adaptation")
    if adaptation_map is not None:
        adaptation: dict[str, Any] = dict(adaptation_map)

        tiling_raw = adaptation.get("tiling", None)
        tiling_map = _optional_mapping(tiling_raw, name="adaptation.tiling")
        if tiling_map is not None:
            tiling: dict[str, Any] = dict(tiling_map)

            if "tile_size" in tiling and tiling["tile_size"] is not None:
                tile_size = _coerce_int(tiling["tile_size"], name="adaptation.tiling.tile_size")
                if tile_size <= 0:
                    raise ValueError("infer-config adaptation.tiling.tile_size must be > 0.")
                tiling["tile_size"] = int(tile_size)

            if "stride" in tiling and tiling["stride"] is not None:
                stride = _coerce_int(tiling["stride"], name="adaptation.tiling.stride")
                if stride <= 0:
                    raise ValueError("infer-config adaptation.tiling.stride must be > 0.")
                tiling["stride"] = int(stride)

            if "score_topk" in tiling and tiling["score_topk"] is not None:
                tiling["score_topk"] = _coerce_float(
                    tiling["score_topk"], name="adaptation.tiling.score_topk"
                )

            if "score_reduce" in tiling and tiling["score_reduce"] is not None:
                mode = str(tiling["score_reduce"]).lower().strip()
                if mode not in _ALLOWED_TILING_SCORE_REDUCE:
                    raise ValueError(
                        "infer-config adaptation.tiling.score_reduce must be one of: "
                        f"{sorted(_ALLOWED_TILING_SCORE_REDUCE)}"
                    )
                tiling["score_reduce"] = mode

            if "map_reduce" in tiling and tiling["map_reduce"] is not None:
                mode = str(tiling["map_reduce"]).lower().strip()
                if mode not in _ALLOWED_TILING_MAP_REDUCE:
                    raise ValueError(
                        "infer-config adaptation.tiling.map_reduce must be one of: "
                        f"{sorted(_ALLOWED_TILING_MAP_REDUCE)}"
                    )
                tiling["map_reduce"] = mode

            adaptation["tiling"] = tiling

        if "save_maps" in adaptation and adaptation["save_maps"] is not None:
            adaptation["save_maps"] = _coerce_bool(adaptation["save_maps"], name="adaptation.save_maps")

        post_cfg = adaptation.get("postprocess", None)
        if post_cfg is not None and not isinstance(post_cfg, Mapping):
            raise ValueError(
                f"infer-config key 'adaptation.postprocess' must be a JSON object/dict or null, got {type(post_cfg).__name__}"
            )

        normalized["adaptation"] = adaptation

    defects_raw = normalized.get("defects", None)
    defects_map = _optional_mapping(defects_raw, name="defects")
    if defects_map is not None:
        defects: dict[str, Any] = dict(defects_map)

        if "enabled" in defects and defects["enabled"] is not None:
            defects["enabled"] = _coerce_bool(defects["enabled"], name="defects.enabled")

        if "pixel_threshold" in defects and defects["pixel_threshold"] is not None:
            defects["pixel_threshold"] = _coerce_float(
                defects["pixel_threshold"], name="defects.pixel_threshold"
            )

        if "pixel_threshold_strategy" in defects and defects["pixel_threshold_strategy"] is not None:
            strategy = str(defects["pixel_threshold_strategy"]).lower().strip()
            if strategy not in _ALLOWED_PIXEL_THRESHOLD_STRATEGIES:
                raise ValueError(
                    "infer-config defects.pixel_threshold_strategy must be one of: "
                    f"{sorted(_ALLOWED_PIXEL_THRESHOLD_STRATEGIES)}"
                )
            defects["pixel_threshold_strategy"] = strategy

        if "pixel_normal_quantile" in defects and defects["pixel_normal_quantile"] is not None:
            q = _coerce_float(defects["pixel_normal_quantile"], name="defects.pixel_normal_quantile")
            if not 0.0 < q < 1.0:
                raise ValueError("infer-config defects.pixel_normal_quantile must be in (0,1).")
            defects["pixel_normal_quantile"] = float(q)

        if "mask_format" in defects and defects["mask_format"] is not None:
            fmt = str(defects["mask_format"]).lower().strip()
            if fmt not in _ALLOWED_MASK_FORMATS:
                raise ValueError(
                    f"infer-config defects.mask_format must be one of: {sorted(_ALLOWED_MASK_FORMATS)}"
                )
            defects["mask_format"] = fmt

        if "roi_xyxy_norm" in defects and defects["roi_xyxy_norm"] is not None:
            roi = defects["roi_xyxy_norm"]
            if not isinstance(roi, (list, tuple)) or len(roi) != 4:
                raise ValueError("infer-config defects.roi_xyxy_norm must be a list of length 4 or null")
            defects["roi_xyxy_norm"] = [float(v) for v in roi]

        if "border_ignore_px" in defects and defects["border_ignore_px"] is not None:
            v = _coerce_int(defects["border_ignore_px"], name="defects.border_ignore_px")
            if v < 0:
                raise ValueError("infer-config defects.border_ignore_px must be >= 0.")
            defects["border_ignore_px"] = int(v)

        ms_raw = defects.get("map_smoothing", None)
        ms_map = _optional_mapping(ms_raw, name="defects.map_smoothing")
        if ms_map is not None:
            ms: dict[str, Any] = dict(ms_map)
            if "method" in ms and ms["method"] is not None:
                method = str(ms["method"]).lower().strip()
                if method not in _ALLOWED_MAP_SMOOTHING_METHODS:
                    raise ValueError(
                        "infer-config defects.map_smoothing.method must be one of: "
                        f"{sorted(_ALLOWED_MAP_SMOOTHING_METHODS)}"
                    )
                ms["method"] = method
            if "ksize" in ms and ms["ksize"] is not None:
                k = _coerce_int(ms["ksize"], name="defects.map_smoothing.ksize")
                if k < 0:
                    raise ValueError("infer-config defects.map_smoothing.ksize must be >= 0.")
                ms["ksize"] = int(k)
            if "sigma" in ms and ms["sigma"] is not None:
                s = _coerce_float(ms["sigma"], name="defects.map_smoothing.sigma")
                if s < 0.0:
                    raise ValueError("infer-config defects.map_smoothing.sigma must be >= 0.")
                ms["sigma"] = float(s)
            defects["map_smoothing"] = ms

        hyst_raw = defects.get("hysteresis", None)
        hyst_map = _optional_mapping(hyst_raw, name="defects.hysteresis")
        if hyst_map is not None:
            hyst: dict[str, Any] = dict(hyst_map)
            if "enabled" in hyst and hyst["enabled"] is not None:
                hyst["enabled"] = _coerce_bool(hyst["enabled"], name="defects.hysteresis.enabled")
            if "low" in hyst:
                hyst["low"] = _coerce_optional_float(hyst["low"], name="defects.hysteresis.low")
            if "high" in hyst:
                hyst["high"] = _coerce_optional_float(hyst["high"], name="defects.hysteresis.high")
            defects["hysteresis"] = hyst

        normalized["defects"] = defects

    # Optional thresholds/checkpoint metadata. Validate types when present.
    if "threshold" in normalized and normalized["threshold"] is not None:
        normalized["threshold"] = _coerce_float(normalized["threshold"], name="threshold")
    if "threshold_provenance" in normalized and normalized["threshold_provenance"] is not None:
        _require_mapping(normalized["threshold_provenance"], name="threshold_provenance")

    if "from_run" in normalized and normalized["from_run"] is not None:
        normalized["from_run"] = str(normalized["from_run"])

    resolved_ckpt: Path | None = None
    if bool(check_files):
        if config_path is None:
            warnings.append("check_files=True but config_path is missing; checkpoint existence was not checked.")
        else:
            resolved_ckpt = resolve_infer_checkpoint_path(normalized, config_path=config_path)

    return InferConfigValidation(payload=normalized, resolved_checkpoint_path=resolved_ckpt, warnings=warnings)


def validate_infer_config_file(
    path: str | Path,
    *,
    category: str | None = None,
    check_files: bool = True,
) -> InferConfigValidation:
    """Load + validate an infer-config JSON file from disk."""

    p = Path(path)
    payload = load_infer_config(p)
    return validate_infer_config_payload(payload, config_path=p, category=category, check_files=check_files)
