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


def prepare_infer_runtime_plan(
    request: InferRuntimePlanRequest,
    *,
    run_inference_impl: Callable[..., Any] | None = None,
) -> InferRuntimePlanResult:
    from pyimgano.defects.pixel_threshold import resolve_pixel_threshold
    from pyimgano.postprocess.anomaly_map import AnomalyMapPostprocess
    from pyimgano.services.inference_service import run_inference

    include_maps = bool(request.include_maps_requested) or bool(
        request.include_maps_by_default
    ) or bool(request.defects_enabled)

    postprocess = None
    if include_maps:
        if bool(request.postprocess_requested):
            postprocess = AnomalyMapPostprocess()
        elif request.infer_config_postprocess is not None:
            postprocess = workbench_adaptation_service.build_postprocess_from_payload(
                request.infer_config_postprocess
            )

    pixel_threshold_value: float | None = None
    pixel_threshold_provenance: dict[str, Any] | None = None
    if bool(request.defects_enabled):
        infer_cfg_source = request.defects_payload_source or "infer_config"
        strategy = str(request.pixel_threshold_strategy)

        infer_cfg_thr = None
        if request.defects_payload is not None:
            raw_thr = request.defects_payload.get("pixel_threshold", None)
            if raw_thr is not None:
                infer_cfg_thr = float(raw_thr)

        calibration_maps = None
        infer_cfg_thr_for_resolve = infer_cfg_thr
        train_paths = list(request.train_paths or [])

        if request.pixel_threshold is None and strategy == "normal_pixel_quantile":
            if not train_paths:
                if infer_cfg_thr is None:
                    raise ValueError(
                        "--defects requires a pixel threshold.\n"
                        "Provide --pixel-threshold, set defects.pixel_threshold in infer_config.json, "
                        "or provide --train-dir for normal-pixel quantile calibration."
                    )
            else:
                infer_cfg_thr_for_resolve = None
                run_inference_fn = run_inference_impl or run_inference
                calibration_maps = []
                calibration_run = run_inference_fn(
                    detector=request.detector,
                    inputs=train_paths,
                    include_maps=True,
                    postprocess=postprocess,
                    batch_size=request.batch_size,
                    amp=bool(request.amp),
                )
                for result in calibration_run.records:
                    if result.anomaly_map is not None:
                        calibration_maps.append(result.anomaly_map)

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

    return InferRuntimePlanResult(
        include_maps=bool(include_maps),
        postprocess=postprocess,
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
