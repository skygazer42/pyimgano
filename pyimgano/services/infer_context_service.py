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
    defects_payload_source: str | None
    illumination_contrast_knobs: Any | None
    tiling_payload: dict[str, Any] | None
    infer_config_postprocess: dict[str, Any] | None
    enable_maps_by_default: bool = False
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


def prepare_from_run_context(request: FromRunInferContextRequest) -> ConfigBackedInferContext:
    cfg = workbench_run_service.load_workbench_config_from_run(request.run_dir)
    report = workbench_run_service.load_report_from_run(request.run_dir)
    _category_name, category_report = workbench_run_service.select_category_report(
        report,
        category=(str(request.from_run_category) if request.from_run_category is not None else None),
    )

    threshold = workbench_run_service.extract_threshold(category_report)
    trained_checkpoint_path = workbench_run_service.resolve_checkpoint_path(
        request.run_dir,
        category_report,
    )

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

    warnings: tuple[str, ...] = ()
    if request.checkpoint_path is not None:
        checkpoint_path = str(request.checkpoint_path)
    else:
        checkpoint_path, warnings = _resolve_from_run_model_checkpoint_path(
            run_dir=str(request.run_dir),
            configured_checkpoint_path=(
                str(cfg.model.checkpoint_path) if cfg.model.checkpoint_path is not None else None
            ),
        )

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

    return ConfigBackedInferContext(
        model_name=str(cfg.model.name),
        preset=(str(preset) if preset is not None else None),
        device=str(device),
        contamination=float(contamination),
        pretrained=bool(pretrained),
        base_user_kwargs=base_user_kwargs,
        checkpoint_path=checkpoint_path,
        trained_checkpoint_path=(
            str(trained_checkpoint_path) if trained_checkpoint_path is not None else None
        ),
        threshold=(float(threshold) if threshold is not None else None),
        defects_payload=asdict(cfg.defects),
        defects_payload_source="from_run",
        illumination_contrast_knobs=illumination_contrast_knobs,
        tiling_payload=tiling_payload,
        infer_config_postprocess=None,
        enable_maps_by_default=False,
        warnings=warnings,
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

    defects_payload = payload.get("defects", None)
    if defects_payload is not None:
        if not isinstance(defects_payload, dict):
            raise ValueError("infer-config key 'defects' must be a JSON object/dict.")
        defects_payload = dict(defects_payload)

    illumination_contrast_knobs = None
    preprocessing_payload = payload.get("preprocessing", None)
    if preprocessing_payload is not None:
        if not isinstance(preprocessing_payload, dict):
            raise ValueError("infer-config key 'preprocessing' must be a JSON object/dict.")
        illumination_payload = preprocessing_payload.get("illumination_contrast", None)
        if illumination_payload is not None:
            if not isinstance(illumination_payload, dict):
                raise ValueError(
                    "infer-config key 'preprocessing.illumination_contrast' must be a JSON object/dict."
                )
            illumination_contrast_knobs = parse_illumination_contrast_knobs(illumination_payload)

    model_payload = payload.get("model", None)
    if not isinstance(model_payload, dict):
        raise ValueError("infer-config must contain a JSON object at key 'model'.")
    model_name = model_payload.get("name", None)
    if model_name is None:
        raise ValueError("infer-config model.name is required.")

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

    threshold = workbench_run_service.extract_threshold(payload)
    trained_checkpoint_path = resolve_infer_checkpoint_path(payload, config_path=config_path)

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

    return ConfigBackedInferContext(
        model_name=str(model_name),
        preset=(str(preset) if preset is not None else None),
        device=str(device),
        contamination=float(contamination),
        pretrained=bool(pretrained),
        base_user_kwargs=base_user_kwargs,
        checkpoint_path=checkpoint_path,
        trained_checkpoint_path=(
            str(trained_checkpoint_path) if trained_checkpoint_path is not None else None
        ),
        threshold=(float(threshold) if threshold is not None else None),
        defects_payload=(dict(defects_payload) if defects_payload is not None else None),
        defects_payload_source=("infer_config" if defects_payload is not None else None),
        illumination_contrast_knobs=illumination_contrast_knobs,
        tiling_payload=tiling_payload,
        infer_config_postprocess=infer_config_postprocess,
        enable_maps_by_default=bool(
            adaptation_payload.get("save_maps", False) or infer_config_postprocess is not None
        ),
        warnings=tuple(str(warning) for warning in schema_warnings),
    )


__all__ = [
    "ConfigBackedInferContext",
    "FromRunInferContextRequest",
    "InferConfigContextRequest",
    "prepare_from_run_context",
    "prepare_infer_config_context",
]
