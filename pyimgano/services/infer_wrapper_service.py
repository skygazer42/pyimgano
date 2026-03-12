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


def apply_infer_detector_wrappers(
    request: InferDetectorWrapperRequest,
) -> InferDetectorWrapperResult:
    detector = request.detector
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

    if tile_size is not None:
        from pyimgano.inference.tiling import TiledDetector

        detector = TiledDetector(
            detector=detector,
            tile_size=int(tile_size),
            stride=(int(tile_stride) if tile_stride is not None else None),
            score_reduce=str(tile_score_reduce),
            score_topk=float(tile_score_topk),
            map_reduce=str(tile_map_reduce),
            u16_max=(int(request.u16_max) if request.u16_max is not None else None),
        )
        if request.threshold is not None:
            setattr(detector, "threshold_", float(request.threshold))

    if request.illumination_contrast_knobs is not None:
        _require_numpy_model_for_preprocessing(str(request.model_name))
        from pyimgano.inference.preprocessing import PreprocessingDetector

        detector = PreprocessingDetector(
            detector=detector,
            illumination_contrast=request.illumination_contrast_knobs,
            u16_max=(int(request.u16_max) if request.u16_max is not None else None),
        )
        if request.threshold is not None:
            setattr(detector, "threshold_", float(request.threshold))

    return InferDetectorWrapperResult(detector=detector)


__all__ = [
    "InferDetectorWrapperRequest",
    "InferDetectorWrapperResult",
    "apply_infer_detector_wrappers",
]
