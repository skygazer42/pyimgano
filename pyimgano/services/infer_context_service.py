from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pyimgano.services.workbench_run_service as workbench_run_service


@dataclass(frozen=True)
class ConfigBackedInferContext:
    model_name: str
    preset: str | None
    device: str
    contamination: float
    pretrained: bool
    base_user_kwargs: dict[str, Any]
    checkpoint_path: str | None
    trained_checkpoint_path: str | None
    threshold: float | None
    defects_payload: dict[str, Any] | None
    prediction_payload: dict[str, Any] | None
    defects_payload_source: str | None
    illumination_contrast_knobs: Any | None
    tiling_payload: dict[str, Any] | None
    infer_config_postprocess: dict[str, Any] | None
    enable_maps_by_default: bool = False
    postprocess_summary: dict[str, Any] | None = None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class FromRunInferContextRequest:
    run_dir: str
    from_run_category: str | None = None
    preset: str | None = None
    device: str | None = None
    contamination: float | None = None
    pretrained: bool | None = None
    model_kwargs: dict[str, Any] | None = None
    checkpoint_path: str | None = None


@dataclass(frozen=True)
class InferConfigContextRequest:
    config_path: str
    infer_category: str | None = None
    preset: str | None = None
    device: str | None = None
    contamination: float | None = None
    pretrained: bool | None = None
    model_kwargs: dict[str, Any] | None = None
    checkpoint_path: str | None = None


def _copy_optional_mapping(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if payload is None:
        return None
    return dict(payload)


def _normalize_warnings(warnings: Any) -> tuple[str, ...]:
    return tuple(str(warning) for warning in (warnings or ()))


def _resolve_from_run_model_checkpoint_path(
    *,
    run_dir: str,
    configured_checkpoint_path: str | None,
) -> tuple[str | None, tuple[str, ...]]:
    if configured_checkpoint_path is None:
        return None, ()

    raw = str(configured_checkpoint_path).strip()
    if not raw:
        return None, ()

    path = Path(raw)
    if path.is_absolute():
        if not path.exists():
            raise FileNotFoundError(f"Model checkpoint_path not found: {path}")
        return str(path), ()

    run_path = Path(str(run_dir))
    candidates = [
        (run_path / path).resolve(),
        (run_path / "artifacts" / path).resolve(),
        (run_path / "checkpoints" / path).resolve(),
    ]

    for candidate in candidates:
        if candidate.exists():
            return str(candidate), ()

    cwd_candidate = path.resolve()
    if cwd_candidate.exists():
        return (
            str(cwd_candidate),
            (
                "model.checkpoint_path resolved relative to CWD; "
                "consider making it relative to --from-run for portability.",
            ),
        )

    tried = "\n".join(f"- {candidate}" for candidate in (candidates + [cwd_candidate]))
    raise FileNotFoundError(
        "Model checkpoint_path not found for --from-run.\n"
        f"model.checkpoint_path={raw!r}\n"
        f"from_run={run_path}\n"
        "Tried:\n"
        f"{tried}"
    )


def _apply_from_run_overrides(
    cfg: Any,
    request: FromRunInferContextRequest,
) -> tuple[str | None, str, float, bool, dict[str, Any]]:
    preset = cfg.model.preset
    if request.preset is not None:
        preset = str(request.preset)

    device = str(cfg.model.device)
    if request.device is not None:
        device = str(request.device)

    contamination = float(cfg.model.contamination)
    if request.contamination is not None:
        contamination = float(request.contamination)

    pretrained = bool(cfg.model.pretrained)
    if request.pretrained is not None:
        pretrained = bool(request.pretrained)

    base_user_kwargs = dict(cfg.model.model_kwargs)
    if request.model_kwargs:
        base_user_kwargs.update(dict(request.model_kwargs))

    return (
        (str(preset) if preset is not None else None),
        str(device),
        float(contamination),
        bool(pretrained),
        base_user_kwargs,
    )


def _resolve_from_run_checkpoint(
    cfg: Any,
    request: FromRunInferContextRequest,
) -> tuple[str | None, tuple[str, ...]]:
    if request.checkpoint_path is not None:
        return str(request.checkpoint_path), ()
    return _resolve_from_run_model_checkpoint_path(
        run_dir=str(request.run_dir),
        configured_checkpoint_path=(
            str(cfg.model.checkpoint_path) if cfg.model.checkpoint_path is not None else None
        ),
    )


def _extract_from_run_optional_payloads(cfg: Any) -> tuple[Any | None, dict[str, Any] | None]:
    illumination_contrast_knobs = None
    try:
        illumination_contrast_knobs = cfg.preprocessing.illumination_contrast
    except Exception:
        illumination_contrast_knobs = None

    tiling_payload: dict[str, Any] | None = None
    try:
        tiling = cfg.adaptation.tiling
    except Exception:
        tiling = None
    if tiling is not None and getattr(tiling, "tile_size", None) is not None:
        tiling_payload = {
            "tile_size": int(tiling.tile_size),
            "stride": int(tiling.stride) if getattr(tiling, "stride", None) is not None else None,
            "score_reduce": str(tiling.score_reduce),
            "score_topk": float(tiling.score_topk),
            "map_reduce": str(tiling.map_reduce),
        }

    return illumination_contrast_knobs, tiling_payload


def _build_config_backed_context(context_payload: dict[str, Any]) -> ConfigBackedInferContext:
    return ConfigBackedInferContext(
        model_name=str(context_payload["model_name"]),
        preset=(
            str(context_payload["preset"]) if context_payload.get("preset") is not None else None
        ),
        device=str(context_payload["device"]),
        contamination=float(context_payload["contamination"]),
        pretrained=bool(context_payload["pretrained"]),
        base_user_kwargs=dict(context_payload["base_user_kwargs"]),
        checkpoint_path=context_payload.get("checkpoint_path"),
        trained_checkpoint_path=(
            str(context_payload["trained_checkpoint_path"])
            if context_payload.get("trained_checkpoint_path") is not None
            else None
        ),
        threshold=(
            float(context_payload["threshold"])
            if context_payload.get("threshold") is not None
            else None
        ),
        defects_payload=_copy_optional_mapping(context_payload.get("defects_payload")),
        prediction_payload=_copy_optional_mapping(context_payload.get("prediction_payload")),
        defects_payload_source=(
            str(context_payload["defects_payload_source"])
            if context_payload.get("defects_payload_source") is not None
            else None
        ),
        illumination_contrast_knobs=context_payload.get("illumination_contrast_knobs"),
        tiling_payload=_copy_optional_mapping(context_payload.get("tiling_payload")),
        infer_config_postprocess=_copy_optional_mapping(context_payload.get("infer_config_postprocess")),
        enable_maps_by_default=bool(context_payload.get("enable_maps_by_default", False)),
        postprocess_summary=_copy_optional_mapping(context_payload.get("postprocess_summary")),
        warnings=_normalize_warnings(context_payload.get("warnings", ())),
    )


def _build_postprocess_summary(
    *,
    defects_payload: dict[str, Any] | None,
    defects_payload_source: str | None,
    prediction_payload: dict[str, Any] | None,
    tiling_payload: dict[str, Any] | None,
    infer_config_postprocess: dict[str, Any] | None,
    enable_maps_by_default: bool,
) -> dict[str, Any]:
    pixel_threshold_strategy = None
    pixel_threshold_in_payload = False
    if isinstance(defects_payload, dict):
        pixel_threshold_strategy = defects_payload.get("pixel_threshold_strategy", None)
        pixel_threshold_in_payload = defects_payload.get("pixel_threshold", None) is not None

    prediction_summary = _copy_optional_mapping(prediction_payload)
    tiling_summary = _tiling_summary(tiling_payload)
    map_postprocess_summary = _map_postprocess_summary(infer_config_postprocess)

    return {
        "has_defects_payload": defects_payload is not None,
        "defects_payload_source": defects_payload_source,
        "pixel_threshold_in_payload": bool(pixel_threshold_in_payload),
        "pixel_threshold_strategy": (
            str(pixel_threshold_strategy) if pixel_threshold_strategy is not None else None
        ),
        "has_prediction_policy": prediction_payload is not None,
        "prediction_policy": prediction_summary,
        "has_tiling": tiling_payload is not None,
        "tiling_summary": tiling_summary,
        "has_map_postprocess": infer_config_postprocess is not None,
        "map_postprocess_summary": map_postprocess_summary,
        "maps_enabled_by_default": bool(enable_maps_by_default),
    }


def _tiling_summary(tiling_payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(tiling_payload, dict):
        return None
    return {
        "tile_size": (
            int(tiling_payload["tile_size"])
            if tiling_payload.get("tile_size") is not None
            else None
        ),
        "stride": (
            int(tiling_payload["stride"]) if tiling_payload.get("stride") is not None else None
        ),
        "score_reduce": (
            str(tiling_payload["score_reduce"])
            if tiling_payload.get("score_reduce") is not None
            else None
        ),
        "score_topk": (
            float(tiling_payload["score_topk"])
            if tiling_payload.get("score_topk") is not None
            else None
        ),
        "map_reduce": (
            str(tiling_payload["map_reduce"])
            if tiling_payload.get("map_reduce") is not None
            else None
        ),
    }


def _map_postprocess_summary(
    infer_config_postprocess: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(infer_config_postprocess, dict):
        return None
    percentile_range = infer_config_postprocess.get("percentile_range", None)
    return {
        "normalize": bool(infer_config_postprocess.get("normalize", False)),
        "normalize_method": (
            str(infer_config_postprocess["normalize_method"])
            if infer_config_postprocess.get("normalize_method") is not None
            else None
        ),
        "percentile_range": (
            [float(item) for item in percentile_range]
            if isinstance(percentile_range, (list, tuple))
            else None
        ),
        "gaussian_sigma": (
            float(infer_config_postprocess["gaussian_sigma"])
            if infer_config_postprocess.get("gaussian_sigma") is not None
            else None
        ),
        "morph_open_ksize": (
            int(infer_config_postprocess["morph_open_ksize"])
            if infer_config_postprocess.get("morph_open_ksize") is not None
            else None
        ),
        "morph_close_ksize": (
            int(infer_config_postprocess["morph_close_ksize"])
            if infer_config_postprocess.get("morph_close_ksize") is not None
            else None
        ),
        "component_threshold": (
            float(infer_config_postprocess["component_threshold"])
            if infer_config_postprocess.get("component_threshold") is not None
            else None
        ),
        "min_component_area": (
            int(infer_config_postprocess["min_component_area"])
            if infer_config_postprocess.get("min_component_area") is not None
            else None
        ),
    }


def prepare_from_run_context(request: FromRunInferContextRequest) -> ConfigBackedInferContext:
    cfg = workbench_run_service.load_workbench_config_from_run(request.run_dir)
    report = workbench_run_service.load_report_from_run(request.run_dir)
    _category_name, category_report = workbench_run_service.select_category_report(
        report,
        category=(
            str(request.from_run_category) if request.from_run_category is not None else None
        ),
    )

    threshold = workbench_run_service.extract_threshold(category_report)
    trained_checkpoint_path = workbench_run_service.resolve_checkpoint_path(
        request.run_dir,
        category_report,
    )

    preset, device, contamination, pretrained, base_user_kwargs = _apply_from_run_overrides(
        cfg,
        request,
    )
    checkpoint_path, warnings = _resolve_from_run_checkpoint(cfg, request)
    illumination_contrast_knobs, tiling_payload = _extract_from_run_optional_payloads(cfg)
    prediction_payload = None
    try:
        prediction_payload = _normalize_prediction_payload(asdict(cfg.prediction))
    except Exception:
        prediction_payload = None

    return _build_config_backed_context(
        {
            "model_name": str(cfg.model.name),
            "preset": preset,
            "device": device,
            "contamination": contamination,
            "pretrained": pretrained,
            "base_user_kwargs": base_user_kwargs,
            "checkpoint_path": checkpoint_path,
            "trained_checkpoint_path": trained_checkpoint_path,
            "threshold": threshold,
            "defects_payload": asdict(cfg.defects),
            "prediction_payload": prediction_payload,
            "defects_payload_source": "from_run",
            "illumination_contrast_knobs": illumination_contrast_knobs,
            "tiling_payload": tiling_payload,
            "infer_config_postprocess": None,
            "enable_maps_by_default": False,
            "postprocess_summary": _build_postprocess_summary(
                defects_payload=asdict(cfg.defects),
                defects_payload_source="from_run",
                prediction_payload=prediction_payload,
                tiling_payload=tiling_payload,
                infer_config_postprocess=None,
                enable_maps_by_default=False,
            ),
            "warnings": warnings,
        }
    )


def _extract_defects_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    defects_payload = payload.get("defects", None)
    if defects_payload is None:
        return None
    if not isinstance(defects_payload, dict):
        raise ValueError("infer-config key 'defects' must be a JSON object/dict.")
    return dict(defects_payload)


def _extract_postprocess_contract_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    postprocess_payload = payload.get("postprocess", None)
    if postprocess_payload is None:
        return None
    if not isinstance(postprocess_payload, dict):
        raise ValueError("infer-config key 'postprocess' must be a JSON object/dict.")
    return dict(postprocess_payload)


def _merge_postprocess_contract_defects_payload(
    defects_payload: dict[str, Any] | None,
    *,
    postprocess_contract_payload: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(postprocess_contract_payload, dict):
        return defects_payload

    pixel_threshold_payload = postprocess_contract_payload.get("pixel_threshold", None)
    if pixel_threshold_payload is None:
        return defects_payload
    if not isinstance(pixel_threshold_payload, dict):
        raise ValueError(
            "infer-config key 'postprocess.pixel_threshold' must be a JSON object/dict."
        )

    merged = dict(defects_payload or {})
    scalar_mappings = (
        ("enabled", "enabled", bool),
        ("strategy", "pixel_threshold_strategy", str),
        ("threshold", "pixel_threshold", float),
        ("normal_quantile", "pixel_normal_quantile", float),
    )
    for source_key, target_key, cast_value in scalar_mappings:
        if source_key not in pixel_threshold_payload:
            continue
        value = pixel_threshold_payload.get(source_key, None)
        merged[target_key] = cast_value(value) if value is not None else None
    return merged if merged else None


def _normalize_prediction_payload(
    prediction_payload: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if prediction_payload is None:
        return None

    reject_confidence_below = prediction_payload.get("reject_confidence_below", None)
    reject_label = prediction_payload.get("reject_label", None)
    if reject_confidence_below is None and reject_label is None:
        return None

    return {
        "reject_confidence_below": (
            float(reject_confidence_below) if reject_confidence_below is not None else None
        ),
        "reject_label": (int(reject_label) if reject_label is not None else None),
    }


def _extract_prediction_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    prediction_payload = payload.get("prediction", None)
    if prediction_payload is None:
        return None
    if not isinstance(prediction_payload, dict):
        raise ValueError("infer-config key 'prediction' must be a JSON object/dict.")
    return _normalize_prediction_payload(dict(prediction_payload))


def _merge_postprocess_contract_prediction_payload(
    prediction_payload: dict[str, Any] | None,
    *,
    postprocess_contract_payload: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(postprocess_contract_payload, dict):
        return prediction_payload

    review_policy_payload = postprocess_contract_payload.get("review_policy", None)
    if review_policy_payload is None:
        return prediction_payload
    if not isinstance(review_policy_payload, dict):
        raise ValueError("infer-config key 'postprocess.review_policy' must be a JSON object/dict.")

    merged = dict(prediction_payload or {})
    if "reject_confidence_below" in review_policy_payload:
        merged["reject_confidence_below"] = review_policy_payload.get(
            "reject_confidence_below", None
        )
    if "reject_label" in review_policy_payload:
        merged["reject_label"] = review_policy_payload.get("reject_label", None)
    return _normalize_prediction_payload(merged)


def _extract_illumination_contrast_knobs(
    payload: dict[str, Any],
    *,
    parse_knobs: Any,
) -> Any | None:
    preprocessing_payload = payload.get("preprocessing", None)
    if preprocessing_payload is None:
        return None
    if not isinstance(preprocessing_payload, dict):
        raise ValueError("infer-config key 'preprocessing' must be a JSON object/dict.")

    illumination_payload = preprocessing_payload.get("illumination_contrast", None)
    if illumination_payload is None:
        return None
    if not isinstance(illumination_payload, dict):
        raise ValueError(
            "infer-config key 'preprocessing.illumination_contrast' must be a JSON object/dict."
        )
    return parse_knobs(illumination_payload)


def _extract_model_payload(payload: dict[str, Any]) -> dict[str, Any]:
    model_payload = payload.get("model", None)
    if not isinstance(model_payload, dict):
        raise ValueError("infer-config must contain a JSON object at key 'model'.")
    model_name = model_payload.get("name", None)
    if model_name is None:
        raise ValueError("infer-config model.name is required.")
    return model_payload


def _extract_adaptation_payload(
    payload: dict[str, Any],
    *,
    postprocess_contract_payload: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]:
    adaptation_payload = payload.get("adaptation", None)
    if adaptation_payload is None:
        adaptation_payload = {}
    if not isinstance(adaptation_payload, dict):
        raise ValueError("infer-config key 'adaptation' must be a JSON object/dict.")

    tiling_payload = None
    tiling_raw = adaptation_payload.get("tiling", None)
    if isinstance(tiling_raw, dict):
        tiling_payload = dict(tiling_raw)

    infer_config_postprocess = None
    postprocess_payload = adaptation_payload.get("postprocess", None)
    if isinstance(postprocess_payload, dict):
        infer_config_postprocess = dict(postprocess_payload)

    if isinstance(postprocess_contract_payload, dict):
        map_postprocess_payload = postprocess_contract_payload.get("map_postprocess", None)
        if map_postprocess_payload is not None:
            if not isinstance(map_postprocess_payload, dict):
                raise ValueError(
                    "infer-config key 'postprocess.map_postprocess' must be a JSON object/dict."
                )
            infer_config_postprocess = dict(map_postprocess_payload)

    return adaptation_payload, tiling_payload, infer_config_postprocess


def _apply_infer_config_overrides(
    *,
    payload: dict[str, Any],
    model_payload: dict[str, Any],
    request: InferConfigContextRequest,
    config_path: Path,
    resolve_infer_model_checkpoint_path: Any,
) -> tuple[str | None, str, float, bool, dict[str, Any], str | None]:
    preset = model_payload.get("preset", None)
    if request.preset is not None:
        preset = str(request.preset)

    device = model_payload.get("device", "cpu")
    if request.device is not None:
        device = str(request.device)

    contamination = model_payload.get("contamination", 0.1)
    if request.contamination is not None:
        contamination = float(request.contamination)

    pretrained = model_payload.get("pretrained", True)
    if request.pretrained is not None:
        pretrained = bool(request.pretrained)

    base_user_kwargs = dict(model_payload.get("model_kwargs", {}) or {})
    if request.model_kwargs:
        base_user_kwargs.update(dict(request.model_kwargs))

    if request.checkpoint_path is not None:
        checkpoint_path = str(request.checkpoint_path)
    elif model_payload.get("checkpoint_path", None) is not None:
        resolved_model_ckpt = resolve_infer_model_checkpoint_path(payload, config_path=config_path)
        checkpoint_path = str(resolved_model_ckpt) if resolved_model_ckpt is not None else None
    else:
        checkpoint_path = None

    return (
        (str(preset) if preset is not None else None),
        str(device),
        float(contamination),
        bool(pretrained),
        base_user_kwargs,
        checkpoint_path,
    )


def prepare_infer_config_context(request: InferConfigContextRequest) -> ConfigBackedInferContext:
    from pyimgano.inference.config import (
        load_infer_config,
        normalize_infer_config_schema,
        resolve_infer_checkpoint_path,
        resolve_infer_model_checkpoint_path,
        select_infer_category,
    )
    from pyimgano.inference.preprocessing import parse_illumination_contrast_knobs

    config_path = Path(str(request.config_path))
    payload = load_infer_config(config_path)
    payload, schema_warnings = normalize_infer_config_schema(payload)
    payload = select_infer_category(
        payload,
        category=(str(request.infer_category) if request.infer_category is not None else None),
    )

    postprocess_contract_payload = _extract_postprocess_contract_payload(payload)
    defects_payload = _extract_defects_payload(payload)
    defects_payload = _merge_postprocess_contract_defects_payload(
        defects_payload,
        postprocess_contract_payload=postprocess_contract_payload,
    )
    prediction_payload = _extract_prediction_payload(payload)
    prediction_payload = _merge_postprocess_contract_prediction_payload(
        prediction_payload,
        postprocess_contract_payload=postprocess_contract_payload,
    )
    illumination_contrast_knobs = _extract_illumination_contrast_knobs(
        payload,
        parse_knobs=parse_illumination_contrast_knobs,
    )
    model_payload = _extract_model_payload(payload)
    model_name = model_payload.get("name", None)
    adaptation_payload, tiling_payload, infer_config_postprocess = _extract_adaptation_payload(
        payload,
        postprocess_contract_payload=postprocess_contract_payload,
    )

    threshold = workbench_run_service.extract_threshold(payload)
    trained_checkpoint_path = resolve_infer_checkpoint_path(payload, config_path=config_path)

    (
        preset,
        device,
        contamination,
        pretrained,
        base_user_kwargs,
        checkpoint_path,
    ) = _apply_infer_config_overrides(
        payload=payload,
        model_payload=model_payload,
        request=request,
        config_path=config_path,
        resolve_infer_model_checkpoint_path=resolve_infer_model_checkpoint_path,
    )

    return _build_config_backed_context(
        {
            "model_name": str(model_name),
            "preset": preset,
            "device": device,
            "contamination": contamination,
            "pretrained": pretrained,
            "base_user_kwargs": base_user_kwargs,
            "checkpoint_path": checkpoint_path,
            "trained_checkpoint_path": trained_checkpoint_path,
            "threshold": threshold,
            "defects_payload": defects_payload,
            "prediction_payload": prediction_payload,
            "defects_payload_source": ("infer_config" if defects_payload is not None else None),
            "illumination_contrast_knobs": illumination_contrast_knobs,
            "tiling_payload": tiling_payload,
            "infer_config_postprocess": infer_config_postprocess,
            "enable_maps_by_default": bool(
                adaptation_payload.get("save_maps", False) or infer_config_postprocess is not None
            ),
            "postprocess_summary": _build_postprocess_summary(
                defects_payload=defects_payload,
                defects_payload_source=("infer_config" if defects_payload is not None else None),
                prediction_payload=prediction_payload,
                tiling_payload=tiling_payload,
                infer_config_postprocess=infer_config_postprocess,
                enable_maps_by_default=bool(
                    adaptation_payload.get("save_maps", False)
                    or infer_config_postprocess is not None
                ),
            ),
            "warnings": _normalize_warnings(schema_warnings),
        }
    )


__all__ = [
    "ConfigBackedInferContext",
    "FromRunInferContextRequest",
    "InferConfigContextRequest",
    "prepare_from_run_context",
    "prepare_infer_config_context",
]
