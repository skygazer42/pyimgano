from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import pyimgano.services.workbench_adaptation_service as workbench_adaptation_service


@dataclass(frozen=True)
class InferRuntimePlanRequest:
    detector: Any
    include_maps_requested: bool = False
    include_maps_by_default: bool = False
    postprocess_requested: bool = False
    infer_config_postprocess: dict[str, Any] | None = None
    postprocess_summary: dict[str, Any] | None = None
    defects_enabled: bool = False
    defects_payload: dict[str, Any] | None = None
    defects_payload_source: str | None = None
    pixel_threshold: float | None = None
    pixel_threshold_strategy: str = "normal_pixel_quantile"
    pixel_normal_quantile: float = 0.999
    roi_xyxy_norm: Sequence[float] | None = None
    train_paths: Sequence[str] | None = None
    batch_size: int | None = None
    amp: bool = False


@dataclass(frozen=True)
class InferRuntimePlanResult:
    include_maps: bool
    postprocess: Any | None
    pixel_threshold_value: float | None
    pixel_threshold_provenance: dict[str, Any] | None
    postprocess_summary: dict[str, Any] | None = None


def _resolve_include_maps(request: InferRuntimePlanRequest) -> bool:
    return (
        bool(request.include_maps_requested)
        or bool(request.include_maps_by_default)
        or bool(request.defects_enabled)
    )


def _build_runtime_postprocess(
    request: InferRuntimePlanRequest,
    *,
    include_maps: bool,
) -> Any | None:
    from pyimgano.postprocess.anomaly_map import AnomalyMapPostprocess

    if not include_maps:
        return None
    if bool(request.postprocess_requested):
        return AnomalyMapPostprocess()
    if request.infer_config_postprocess is not None:
        return workbench_adaptation_service.build_postprocess_from_payload(
            request.infer_config_postprocess
        )
    return None


def _extract_infer_config_pixel_threshold(defects_payload: dict[str, Any] | None) -> float | None:
    if defects_payload is None:
        return None
    raw_threshold = defects_payload.get("pixel_threshold", None)
    return float(raw_threshold) if raw_threshold is not None else None


def _collect_calibration_maps(
    request: InferRuntimePlanRequest,
    *,
    postprocess: Any | None,
    run_inference_impl: Callable[..., Any] | None,
) -> list[Any]:
    from pyimgano.services.inference_service import run_inference

    run_inference_fn = run_inference_impl or run_inference
    calibration_maps: list[Any] = []
    calibration_run = run_inference_fn(
        detector=request.detector,
        inputs=list(request.train_paths or []),
        include_maps=True,
        postprocess=postprocess,
        batch_size=request.batch_size,
        amp=bool(request.amp),
    )
    for result in calibration_run.records:
        if result.anomaly_map is not None:
            calibration_maps.append(result.anomaly_map)
    return calibration_maps


def _resolve_threshold_inputs(
    request: InferRuntimePlanRequest,
    *,
    postprocess: Any | None,
    run_inference_impl: Callable[..., Any] | None,
) -> tuple[float | None, list[Any] | None, str]:
    infer_cfg_source = request.defects_payload_source or "infer_config"
    infer_cfg_threshold = _extract_infer_config_pixel_threshold(request.defects_payload)
    train_paths = list(request.train_paths or [])

    if request.pixel_threshold is not None:
        return infer_cfg_threshold, None, str(infer_cfg_source)

    if str(request.pixel_threshold_strategy) != "normal_pixel_quantile":
        return infer_cfg_threshold, None, str(infer_cfg_source)

    if not train_paths:
        if infer_cfg_threshold is None:
            raise ValueError(
                "--defects requires a pixel threshold.\n"
                "Provide --pixel-threshold, set defects.pixel_threshold in infer_config.json, "
                "or provide --train-dir for normal-pixel quantile calibration."
            )
        return infer_cfg_threshold, None, str(infer_cfg_source)

    calibration_maps = _collect_calibration_maps(
        request,
        postprocess=postprocess,
        run_inference_impl=run_inference_impl,
    )
    return None, calibration_maps, str(infer_cfg_source)


def _resolve_include_maps(request: InferRuntimePlanRequest) -> bool:
    return (
        bool(request.include_maps_requested)
        or bool(request.include_maps_by_default)
        or bool(request.defects_enabled)
    )


def _build_runtime_postprocess(
    request: InferRuntimePlanRequest,
    *,
    include_maps: bool,
) -> Any | None:
    from pyimgano.postprocess.anomaly_map import AnomalyMapPostprocess

    if not include_maps:
        return None
    if bool(request.postprocess_requested):
        return AnomalyMapPostprocess()
    if request.infer_config_postprocess is not None:
        return workbench_adaptation_service.build_postprocess_from_payload(
            request.infer_config_postprocess
        )
    return None


def _extract_infer_config_pixel_threshold(defects_payload: dict[str, Any] | None) -> float | None:
    if defects_payload is None:
        return None
    raw_threshold = defects_payload.get("pixel_threshold", None)
    return float(raw_threshold) if raw_threshold is not None else None


def _collect_calibration_maps(
    request: InferRuntimePlanRequest,
    *,
    postprocess: Any | None,
    run_inference_impl: Callable[..., Any] | None,
) -> list[Any]:
    from pyimgano.services.inference_service import run_inference

    run_inference_fn = run_inference_impl or run_inference
    calibration_maps: list[Any] = []
    calibration_run = run_inference_fn(
        detector=request.detector,
        inputs=list(request.train_paths or []),
        include_maps=True,
        postprocess=postprocess,
        batch_size=request.batch_size,
        amp=bool(request.amp),
    )
    for result in calibration_run.records:
        if result.anomaly_map is not None:
            calibration_maps.append(result.anomaly_map)
    return calibration_maps


def _resolve_threshold_inputs(
    request: InferRuntimePlanRequest,
    *,
    postprocess: Any | None,
    run_inference_impl: Callable[..., Any] | None,
) -> tuple[float | None, list[Any] | None, str]:
    infer_cfg_source = request.defects_payload_source or "infer_config"
    infer_cfg_threshold = _extract_infer_config_pixel_threshold(request.defects_payload)
    train_paths = list(request.train_paths or [])

    if request.pixel_threshold is not None:
        return infer_cfg_threshold, None, str(infer_cfg_source)

    if str(request.pixel_threshold_strategy) != "normal_pixel_quantile":
        return infer_cfg_threshold, None, str(infer_cfg_source)

    if not train_paths:
        if infer_cfg_threshold is None:
            raise ValueError(
                "--defects requires a pixel threshold.\n"
                "Provide --pixel-threshold, set defects.pixel_threshold in infer_config.json, "
                "or provide --train-dir for normal-pixel quantile calibration."
            )
        return infer_cfg_threshold, None, str(infer_cfg_source)

    calibration_maps = _collect_calibration_maps(
        request,
        postprocess=postprocess,
        run_inference_impl=run_inference_impl,
    )
    return None, calibration_maps, str(infer_cfg_source)


def prepare_infer_runtime_plan(
    request: InferRuntimePlanRequest,
    *,
    run_inference_impl: Callable[..., Any] | None = None,
) -> InferRuntimePlanResult:
    from pyimgano.defects.pixel_threshold import resolve_pixel_threshold

    include_maps = _resolve_include_maps(request)
    postprocess = _build_runtime_postprocess(request, include_maps=include_maps)

    pixel_threshold_value: float | None = None
    pixel_threshold_provenance: dict[str, Any] | None = None
    if bool(request.defects_enabled):
        infer_cfg_thr_for_resolve, calibration_maps, infer_cfg_source = _resolve_threshold_inputs(
            request,
            postprocess=postprocess,
            run_inference_impl=run_inference_impl,
        )

        pixel_threshold_value, pixel_threshold_provenance = resolve_pixel_threshold(
            pixel_threshold=(
                float(request.pixel_threshold) if request.pixel_threshold is not None else None
            ),
            pixel_threshold_strategy=str(request.pixel_threshold_strategy),
            infer_config_pixel_threshold=infer_cfg_thr_for_resolve,
            calibration_maps=calibration_maps,
            pixel_normal_quantile=float(request.pixel_normal_quantile),
            infer_config_source=str(infer_cfg_source),
            roi_xyxy_norm=(
                list(request.roi_xyxy_norm) if request.roi_xyxy_norm is not None else None
            ),
        )

    summary = None
    if request.postprocess_summary is not None:
        runtime_postprocess_source = None
        if bool(request.postprocess_requested):
            runtime_postprocess_source = "cli"
        elif request.infer_config_postprocess is not None and postprocess is not None:
            runtime_postprocess_source = "infer_config"

        summary = dict(request.postprocess_summary)
        summary.update(
            {
                "maps_requested": bool(request.include_maps_requested),
                "maps_enabled": bool(include_maps),
                "runtime_postprocess_applied": bool(postprocess is not None),
                "runtime_postprocess_source": runtime_postprocess_source,
                "defects_enabled": bool(request.defects_enabled),
                "pixel_threshold_resolved": bool(pixel_threshold_value is not None),
                "pixel_threshold_source": (
                    str(pixel_threshold_provenance.get("source"))
                    if isinstance(pixel_threshold_provenance, dict)
                    and pixel_threshold_provenance.get("source") is not None
                    else None
                ),
            }
        )

    return InferRuntimePlanResult(
        include_maps=bool(include_maps),
        postprocess=postprocess,
        postprocess_summary=summary,
        pixel_threshold_value=(
            float(pixel_threshold_value) if pixel_threshold_value is not None else None
        ),
        pixel_threshold_provenance=(
            dict(pixel_threshold_provenance) if pixel_threshold_provenance is not None else None
        ),
    )


__all__ = [
    "InferRuntimePlanRequest",
    "InferRuntimePlanResult",
    "prepare_infer_runtime_plan",
]
