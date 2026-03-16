from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pyimgano.models.registry import MODEL_REGISTRY


@dataclass(frozen=True)
class InferDetectorWrapperRequest:
    detector: Any
    model_name: str
    threshold: float | None = None
    tiling_payload: dict[str, Any] | None = None
    tile_size: int | None = None
    tile_stride: int | None = None
    tile_score_reduce: str = "max"
    tile_score_topk: float = 0.1
    tile_map_reduce: str = "max"
    illumination_contrast_knobs: Any | None = None
    u16_max: int | None = None


@dataclass(frozen=True)
class InferDetectorWrapperResult:
    detector: Any


@dataclass(frozen=True)
class _ResolvedTilingOptions:
    tile_size: int | None
    tile_stride: int | None
    tile_score_reduce: str
    tile_score_topk: float
    tile_map_reduce: str


def _require_numpy_model_for_preprocessing(model_name: str) -> None:
    from pyimgano.models.capabilities import compute_model_capabilities

    entry = MODEL_REGISTRY.info(str(model_name))
    caps = compute_model_capabilities(entry)
    supported_input_modes = tuple(str(mode) for mode in caps.input_modes)
    if "numpy" in supported_input_modes:
        return

    raise ValueError(
        "PREPROCESSING_REQUIRES_NUMPY_MODEL: preprocessing.illumination_contrast requires a model that supports numpy inputs. "
        f"model={model_name!r} supported_input_modes={supported_input_modes!r}. "
        "Choose a model with tag 'numpy' (e.g. vision_patchcore) or remove preprocessing from infer-config."
    )


def _resolve_tiling_options(request: InferDetectorWrapperRequest) -> _ResolvedTilingOptions:
    tile_size = request.tile_size
    tile_stride = request.tile_stride
    tile_score_reduce = request.tile_score_reduce
    tile_score_topk = request.tile_score_topk
    tile_map_reduce = request.tile_map_reduce

    if (
        tile_size is None
        and isinstance(request.tiling_payload, dict)
        and request.tiling_payload.get("tile_size", None) is not None
    ):
        tile_size = int(request.tiling_payload.get("tile_size"))
        if request.tiling_payload.get("stride", None) is not None:
            tile_stride = int(request.tiling_payload.get("stride"))
        if request.tiling_payload.get("score_reduce", None) is not None:
            tile_score_reduce = str(request.tiling_payload.get("score_reduce"))
        if request.tiling_payload.get("score_topk", None) is not None:
            tile_score_topk = float(request.tiling_payload.get("score_topk"))
        if request.tiling_payload.get("map_reduce", None) is not None:
            tile_map_reduce = str(request.tiling_payload.get("map_reduce"))

    return _ResolvedTilingOptions(
        tile_size=(int(tile_size) if tile_size is not None else None),
        tile_stride=(int(tile_stride) if tile_stride is not None else None),
        tile_score_reduce=str(tile_score_reduce),
        tile_score_topk=float(tile_score_topk),
        tile_map_reduce=str(tile_map_reduce),
    )


def _apply_threshold_if_present(detector: Any, threshold: float | None) -> Any:
    if threshold is not None:
        setattr(detector, "threshold_", float(threshold))
    return detector


def _wrap_with_tiling(
    detector: Any,
    *,
    tiling: _ResolvedTilingOptions,
    threshold: float | None,
    u16_max: int | None,
) -> Any:
    if tiling.tile_size is None:
        return detector

    from pyimgano.inference.tiling import TiledDetector

    wrapped = TiledDetector(
        detector=detector,
        tile_size=int(tiling.tile_size),
        stride=tiling.tile_stride,
        score_reduce=tiling.tile_score_reduce,
        score_topk=tiling.tile_score_topk,
        map_reduce=tiling.tile_map_reduce,
        u16_max=(int(u16_max) if u16_max is not None else None),
    )
    return _apply_threshold_if_present(wrapped, threshold)


def _wrap_with_preprocessing(
    detector: Any,
    *,
    model_name: str,
    illumination_contrast_knobs: Any | None,
    threshold: float | None,
    u16_max: int | None,
) -> Any:
    if illumination_contrast_knobs is None:
        return detector

    _require_numpy_model_for_preprocessing(str(model_name))
    from pyimgano.inference.preprocessing import PreprocessingDetector

    wrapped = PreprocessingDetector(
        detector=detector,
        illumination_contrast=illumination_contrast_knobs,
        u16_max=(int(u16_max) if u16_max is not None else None),
    )
    return _apply_threshold_if_present(wrapped, threshold)


def apply_infer_detector_wrappers(
    request: InferDetectorWrapperRequest,
) -> InferDetectorWrapperResult:
    detector = request.detector
    tiling = _resolve_tiling_options(request)
    detector = _wrap_with_tiling(
        detector,
        tiling=tiling,
        threshold=request.threshold,
        u16_max=request.u16_max,
    )
    detector = _wrap_with_preprocessing(
        detector,
        model_name=request.model_name,
        illumination_contrast_knobs=request.illumination_contrast_knobs,
        threshold=request.threshold,
        u16_max=request.u16_max,
    )

    return InferDetectorWrapperResult(detector=detector)


__all__ = [
    "InferDetectorWrapperRequest",
    "InferDetectorWrapperResult",
    "apply_infer_detector_wrappers",
]
