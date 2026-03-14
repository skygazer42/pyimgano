from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from pyimgano.inference.api import result_to_jsonable


@dataclass(frozen=True)
class DefectsArtifactConfig:
    pixel_threshold_value: float
    pixel_threshold_provenance: dict[str, Any]
    roi_xyxy_norm: Sequence[float] | None = None
    mask_space: str = "roi"
    border_ignore_px: int = 0
    map_smoothing_method: str = "none"
    map_smoothing_ksize: int = 0
    map_smoothing_sigma: float = 0.0
    hysteresis_enabled: bool = False
    hysteresis_low: float | None = None
    hysteresis_high: float | None = None
    open_ksize: int = 0
    close_ksize: int = 0
    fill_holes: bool = False
    mask_dilate_ksize: int = 0
    min_area: int = 0
    min_fill_ratio: float | None = None
    max_aspect_ratio: float | None = None
    min_solidity: float | None = None
    min_score_max: float | None = None
    min_score_mean: float | None = None
    merge_nearby_enabled: bool = False
    merge_nearby_max_gap_px: int = 0
    max_regions_sort_by: str = "score_max"
    max_regions: int | None = None
    masks_dir: str | None = None
    mask_format: str = "png"
    defects_image_space: bool = False


@dataclass(frozen=True)
class DefectsArtifactConfigBuildRequest:
    defects_enabled: bool = False
    pixel_threshold_value: float | None = None
    pixel_threshold_provenance: dict[str, Any] | None = None
    roi_xyxy_norm: Sequence[float] | None = None
    mask_space: str = "roi"
    border_ignore_px: int = 0
    map_smoothing_method: str = "none"
    map_smoothing_ksize: int = 0
    map_smoothing_sigma: float = 0.0
    hysteresis_enabled: bool = False
    hysteresis_low: float | None = None
    hysteresis_high: float | None = None
    open_ksize: int = 0
    close_ksize: int = 0
    fill_holes: bool = False
    mask_dilate_ksize: int = 0
    min_area: int = 0
    min_fill_ratio: float | None = None
    max_aspect_ratio: float | None = None
    min_solidity: float | None = None
    min_score_max: float | None = None
    min_score_mean: float | None = None
    merge_nearby_enabled: bool = False
    merge_nearby_max_gap_px: int = 0
    max_regions_sort_by: str = "score_max"
    max_regions: int | None = None
    masks_dir: str | None = None
    mask_format: str = "png"
    defects_image_space: bool = False


@dataclass(frozen=True)
class InferResultArtifactRequest:
    index: int
    input_path: str
    result: Any
    include_status: bool = False
    include_anomaly_map_values: bool = False
    maps_dir: str | None = None
    overlays_dir: str | None = None
    defects_config: DefectsArtifactConfig | None = None


@dataclass(frozen=True)
class InferResultArtifactBuildRequest:
    index: int
    input_path: str
    result: Any
    include_status: bool = False
    include_anomaly_map_values: bool = False
    maps_dir: str | None = None
    overlays_dir: str | None = None
    defects: DefectsArtifactConfigBuildRequest | None = None


@dataclass(frozen=True)
class InferArtifactOptions:
    include_anomaly_map_values: bool = False
    maps_dir: str | None = None
    overlays_dir: str | None = None
    defects_enabled: bool = False
    pixel_threshold_value: float | None = None
    pixel_threshold_provenance: dict[str, Any] | None = None
    roi_xyxy_norm: Sequence[float] | None = None
    mask_space: str = "roi"
    border_ignore_px: int = 0
    map_smoothing_method: str = "none"
    map_smoothing_ksize: int = 0
    map_smoothing_sigma: float = 0.0
    hysteresis_enabled: bool = False
    hysteresis_low: float | None = None
    hysteresis_high: float | None = None
    open_ksize: int = 0
    close_ksize: int = 0
    fill_holes: bool = False
    mask_dilate_ksize: int = 0
    min_area: int = 0
    min_fill_ratio: float | None = None
    max_aspect_ratio: float | None = None
    min_solidity: float | None = None
    min_score_max: float | None = None
    min_score_mean: float | None = None
    merge_nearby_enabled: bool = False
    merge_nearby_max_gap_px: int = 0
    max_regions_sort_by: str = "score_max"
    max_regions: int | None = None
    masks_dir: str | None = None
    mask_format: str = "png"
    defects_image_space: bool = False


@dataclass(frozen=True)
class InferResultArtifactAssemblyRequest:
    index: int
    input_path: str
    result: Any
    include_status: bool = False
    options: InferArtifactOptions = field(default_factory=InferArtifactOptions)


@dataclass(frozen=True)
class InferResultArtifactCliRequest:
    index: int
    input_path: str
    result: Any
    cli_args: Any
    include_status: bool = False
    maps_dir: str | None = None
    overlays_dir: str | None = None
    masks_dir: str | None = None
    pixel_threshold_value: float | None = None
    pixel_threshold_provenance: dict[str, Any] | None = None


@dataclass(frozen=True)
class InferResultArtifactResult:
    record: dict[str, Any]
    regions_payload: dict[str, Any] | None = None


def build_defects_artifact_config(
    request: DefectsArtifactConfigBuildRequest,
) -> DefectsArtifactConfig | None:
    if not bool(request.defects_enabled):
        return None

    if request.pixel_threshold_value is None or request.pixel_threshold_provenance is None:
        raise RuntimeError("Internal error: pixel threshold was not resolved for --defects.")

    return DefectsArtifactConfig(
        pixel_threshold_value=float(request.pixel_threshold_value),
        pixel_threshold_provenance=dict(request.pixel_threshold_provenance),
        roi_xyxy_norm=(list(request.roi_xyxy_norm) if request.roi_xyxy_norm is not None else None),
        mask_space=str(request.mask_space),
        border_ignore_px=int(request.border_ignore_px),
        map_smoothing_method=str(request.map_smoothing_method),
        map_smoothing_ksize=int(request.map_smoothing_ksize),
        map_smoothing_sigma=float(request.map_smoothing_sigma),
        hysteresis_enabled=bool(request.hysteresis_enabled),
        hysteresis_low=(
            float(request.hysteresis_low) if request.hysteresis_low is not None else None
        ),
        hysteresis_high=(
            float(request.hysteresis_high) if request.hysteresis_high is not None else None
        ),
        open_ksize=int(request.open_ksize),
        close_ksize=int(request.close_ksize),
        fill_holes=bool(request.fill_holes),
        mask_dilate_ksize=int(request.mask_dilate_ksize),
        min_area=int(request.min_area),
        min_fill_ratio=(
            float(request.min_fill_ratio) if request.min_fill_ratio is not None else None
        ),
        max_aspect_ratio=(
            float(request.max_aspect_ratio) if request.max_aspect_ratio is not None else None
        ),
        min_solidity=(float(request.min_solidity) if request.min_solidity is not None else None),
        min_score_max=(
            float(request.min_score_max) if request.min_score_max is not None else None
        ),
        min_score_mean=(
            float(request.min_score_mean) if request.min_score_mean is not None else None
        ),
        merge_nearby_enabled=bool(request.merge_nearby_enabled),
        merge_nearby_max_gap_px=int(request.merge_nearby_max_gap_px),
        max_regions_sort_by=str(request.max_regions_sort_by),
        max_regions=(int(request.max_regions) if request.max_regions is not None else None),
        masks_dir=(str(request.masks_dir) if request.masks_dir is not None else None),
        mask_format=str(request.mask_format),
        defects_image_space=bool(request.defects_image_space),
    )


def build_infer_result_artifact_request(
    request: InferResultArtifactBuildRequest,
) -> InferResultArtifactRequest:
    defects_config = None
    if request.defects is not None:
        defects_config = build_defects_artifact_config(request.defects)

    return InferResultArtifactRequest(
        index=int(request.index),
        input_path=str(request.input_path),
        result=request.result,
        include_status=bool(request.include_status),
        include_anomaly_map_values=bool(request.include_anomaly_map_values),
        maps_dir=(str(request.maps_dir) if request.maps_dir is not None else None),
        overlays_dir=(str(request.overlays_dir) if request.overlays_dir is not None else None),
        defects_config=defects_config,
    )


def build_infer_result_artifact_request_from_options(
    request: InferResultArtifactAssemblyRequest,
) -> InferResultArtifactBuildRequest:
    options = request.options

    return InferResultArtifactBuildRequest(
        index=int(request.index),
        input_path=str(request.input_path),
        result=request.result,
        include_status=bool(request.include_status),
        include_anomaly_map_values=bool(options.include_anomaly_map_values),
        maps_dir=(str(options.maps_dir) if options.maps_dir is not None else None),
        overlays_dir=(str(options.overlays_dir) if options.overlays_dir is not None else None),
        defects=DefectsArtifactConfigBuildRequest(
            defects_enabled=bool(options.defects_enabled),
            pixel_threshold_value=(
                float(options.pixel_threshold_value)
                if options.pixel_threshold_value is not None
                else None
            ),
            pixel_threshold_provenance=(
                dict(options.pixel_threshold_provenance)
                if options.pixel_threshold_provenance is not None
                else None
            ),
            roi_xyxy_norm=(
                list(options.roi_xyxy_norm) if options.roi_xyxy_norm is not None else None
            ),
            mask_space=str(options.mask_space),
            border_ignore_px=int(options.border_ignore_px),
            map_smoothing_method=str(options.map_smoothing_method),
            map_smoothing_ksize=int(options.map_smoothing_ksize),
            map_smoothing_sigma=float(options.map_smoothing_sigma),
            hysteresis_enabled=bool(options.hysteresis_enabled),
            hysteresis_low=(
                float(options.hysteresis_low) if options.hysteresis_low is not None else None
            ),
            hysteresis_high=(
                float(options.hysteresis_high) if options.hysteresis_high is not None else None
            ),
            open_ksize=int(options.open_ksize),
            close_ksize=int(options.close_ksize),
            fill_holes=bool(options.fill_holes),
            mask_dilate_ksize=int(options.mask_dilate_ksize),
            min_area=int(options.min_area),
            min_fill_ratio=(
                float(options.min_fill_ratio) if options.min_fill_ratio is not None else None
            ),
            max_aspect_ratio=(
                float(options.max_aspect_ratio) if options.max_aspect_ratio is not None else None
            ),
            min_solidity=(
                float(options.min_solidity) if options.min_solidity is not None else None
            ),
            min_score_max=(
                float(options.min_score_max) if options.min_score_max is not None else None
            ),
            min_score_mean=(
                float(options.min_score_mean) if options.min_score_mean is not None else None
            ),
            merge_nearby_enabled=bool(options.merge_nearby_enabled),
            merge_nearby_max_gap_px=int(options.merge_nearby_max_gap_px),
            max_regions_sort_by=str(options.max_regions_sort_by),
            max_regions=(int(options.max_regions) if options.max_regions is not None else None),
            masks_dir=(str(options.masks_dir) if options.masks_dir is not None else None),
            mask_format=str(options.mask_format),
            defects_image_space=bool(options.defects_image_space),
        ),
    )


def build_infer_result_artifact_build_request_from_cli(
    request: InferResultArtifactCliRequest,
) -> InferResultArtifactBuildRequest:
    args = request.cli_args
    roi_xyxy_norm = getattr(args, "roi_xyxy_norm", None)
    max_regions = getattr(args, "defect_max_regions", None)
    map_smoothing_sigma_raw = getattr(args, "defect_map_smoothing_sigma", None)
    map_smoothing_sigma = 0.0 if map_smoothing_sigma_raw is None else float(map_smoothing_sigma_raw)

    return build_infer_result_artifact_request_from_options(
        InferResultArtifactAssemblyRequest(
            index=int(request.index),
            input_path=str(request.input_path),
            result=request.result,
            include_status=bool(request.include_status),
            options=InferArtifactOptions(
                include_anomaly_map_values=bool(
                    getattr(args, "include_anomaly_map_values", False)
                ),
                maps_dir=(str(request.maps_dir) if request.maps_dir is not None else None),
                overlays_dir=(
                    str(request.overlays_dir) if request.overlays_dir is not None else None
                ),
                defects_enabled=bool(getattr(args, "defects", False)),
                pixel_threshold_value=(
                    float(request.pixel_threshold_value)
                    if request.pixel_threshold_value is not None
                    else None
                ),
                pixel_threshold_provenance=(
                    dict(request.pixel_threshold_provenance)
                    if request.pixel_threshold_provenance is not None
                    else None
                ),
                roi_xyxy_norm=(list(roi_xyxy_norm) if roi_xyxy_norm is not None else None),
                mask_space=str(getattr(args, "defects_mask_space", "roi")),
                border_ignore_px=int(getattr(args, "defect_border_ignore_px", 0)),
                map_smoothing_method=str(getattr(args, "defect_map_smoothing", "none")),
                map_smoothing_ksize=int(getattr(args, "defect_map_smoothing_ksize", 0)),
                map_smoothing_sigma=map_smoothing_sigma,
                hysteresis_enabled=bool(getattr(args, "defect_hysteresis", False)),
                hysteresis_low=(
                    float(getattr(args, "defect_hysteresis_low"))
                    if getattr(args, "defect_hysteresis_low", None) is not None
                    else None
                ),
                hysteresis_high=(
                    float(getattr(args, "defect_hysteresis_high"))
                    if getattr(args, "defect_hysteresis_high", None) is not None
                    else None
                ),
                open_ksize=int(getattr(args, "defect_open_ksize", 0)),
                close_ksize=int(getattr(args, "defect_close_ksize", 0)),
                fill_holes=bool(getattr(args, "defect_fill_holes", False)),
                mask_dilate_ksize=int(getattr(args, "defects_mask_dilate", 0)),
                min_area=int(getattr(args, "defect_min_area", 0)),
                min_fill_ratio=(
                    float(getattr(args, "defect_min_fill_ratio"))
                    if getattr(args, "defect_min_fill_ratio", None) is not None
                    else None
                ),
                max_aspect_ratio=(
                    float(getattr(args, "defect_max_aspect_ratio"))
                    if getattr(args, "defect_max_aspect_ratio", None) is not None
                    else None
                ),
                min_solidity=(
                    float(getattr(args, "defect_min_solidity"))
                    if getattr(args, "defect_min_solidity", None) is not None
                    else None
                ),
                min_score_max=(
                    float(getattr(args, "defect_min_score_max"))
                    if getattr(args, "defect_min_score_max", None) is not None
                    else None
                ),
                min_score_mean=(
                    float(getattr(args, "defect_min_score_mean"))
                    if getattr(args, "defect_min_score_mean", None) is not None
                    else None
                ),
                merge_nearby_enabled=bool(getattr(args, "defect_merge_nearby", False)),
                merge_nearby_max_gap_px=int(getattr(args, "defect_merge_nearby_max_gap_px", 0)),
                max_regions_sort_by=str(getattr(args, "defect_max_regions_sort_by", "score_max")),
                max_regions=(int(max_regions) if max_regions is not None else None),
                masks_dir=(str(request.masks_dir) if request.masks_dir is not None else None),
                mask_format=str(getattr(args, "mask_format", "png")),
                defects_image_space=bool(getattr(args, "defects_image_space", False)),
            ),
        )
    )


def build_infer_result_artifact_request_from_cli(
    request: InferResultArtifactCliRequest,
) -> InferResultArtifactRequest:
    return build_infer_result_artifact_request(
        build_infer_result_artifact_build_request_from_cli(request)
    )


def _save_anomaly_map(
    *,
    index: int,
    input_path: str,
    anomaly_map: np.ndarray | None,
    maps_dir: str | None,
) -> str | None:
    if maps_dir is None or anomaly_map is None:
        return None

    out_dir = Path(maps_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(input_path).stem
    out_path = out_dir / f"{int(index):06d}_{stem}.npy"
    np.save(out_path, np.asarray(anomaly_map, dtype=np.float32))
    return str(out_path)


def _save_overlay(
    *,
    index: int,
    input_path: str,
    anomaly_map: np.ndarray | None,
    defect_mask: np.ndarray | None,
    overlays_dir: str | None,
) -> None:
    if overlays_dir is None:
        return

    from pyimgano.defects.overlays import save_overlay_image

    out_dir = Path(overlays_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(input_path).stem
    out_path = out_dir / f"{int(index):06d}_{stem}.png"
    save_overlay_image(
        input_path,
        anomaly_map=anomaly_map,
        defect_mask=defect_mask,
        out_path=out_path,
    )


def _project_regions_to_image_space(
    *,
    input_path: str,
    regions: list[dict[str, Any]],
    src_hw: tuple[int, int],
) -> None:
    from PIL import Image

    from pyimgano.defects.space import scale_bbox_xyxy_inclusive

    with Image.open(input_path) as im:
        w_img, h_img = im.size

    dst_hw = (int(h_img), int(w_img))
    for region in regions:
        bbox = region.get("bbox_xyxy", None)
        if bbox is None:
            continue
        region["bbox_xyxy_image"] = scale_bbox_xyxy_inclusive(
            bbox,
            src_hw=src_hw,
            dst_hw=dst_hw,
        )


def materialize_infer_result_artifacts(
    request: InferResultArtifactRequest,
) -> InferResultArtifactResult:
    anomaly_map = (
        None
        if getattr(request.result, "anomaly_map", None) is None
        else np.asarray(request.result.anomaly_map, dtype=np.float32)
    )
    anomaly_map_path = _save_anomaly_map(
        index=int(request.index),
        input_path=str(request.input_path),
        anomaly_map=anomaly_map,
        maps_dir=request.maps_dir,
    )

    record = result_to_jsonable(
        request.result,
        anomaly_map_path=anomaly_map_path,
        include_anomaly_map_values=bool(request.include_anomaly_map_values),
    )
    record["index"] = int(request.index)
    record["input"] = str(request.input_path)
    if bool(request.include_status):
        record["status"] = "ok"

    saved_defects_mask: np.ndarray | None = None
    regions_payload: dict[str, Any] | None = None
    cfg = request.defects_config
    if cfg is not None:
        if anomaly_map is None:
            raise ValueError(
                "Defects export requires anomaly maps, but no anomaly_map was returned.\n"
                "Try a detector that supports get_anomaly_map/predict_anomaly_map, and "
                "ensure --include-maps (or --defects) is enabled."
            )
        if cfg.pixel_threshold_provenance is None:
            raise RuntimeError("Internal error: pixel threshold was not resolved for --defects.")

        from pyimgano.defects.extract import extract_defects_from_anomaly_map
        from pyimgano.defects.io import save_binary_mask

        defects = extract_defects_from_anomaly_map(
            anomaly_map,
            pixel_threshold=float(cfg.pixel_threshold_value),
            roi_xyxy_norm=(list(cfg.roi_xyxy_norm) if cfg.roi_xyxy_norm is not None else None),
            mask_space=str(cfg.mask_space),
            border_ignore_px=int(cfg.border_ignore_px),
            map_smoothing_method=str(cfg.map_smoothing_method),
            map_smoothing_ksize=int(cfg.map_smoothing_ksize),
            map_smoothing_sigma=float(cfg.map_smoothing_sigma),
            hysteresis_enabled=bool(cfg.hysteresis_enabled),
            hysteresis_low=(float(cfg.hysteresis_low) if cfg.hysteresis_low is not None else None),
            hysteresis_high=(
                float(cfg.hysteresis_high) if cfg.hysteresis_high is not None else None
            ),
            open_ksize=int(cfg.open_ksize),
            close_ksize=int(cfg.close_ksize),
            fill_holes=bool(cfg.fill_holes),
            mask_dilate_ksize=int(cfg.mask_dilate_ksize),
            min_area=int(cfg.min_area),
            min_fill_ratio=(
                float(cfg.min_fill_ratio) if cfg.min_fill_ratio is not None else None
            ),
            max_aspect_ratio=(
                float(cfg.max_aspect_ratio) if cfg.max_aspect_ratio is not None else None
            ),
            min_solidity=(float(cfg.min_solidity) if cfg.min_solidity is not None else None),
            min_score_max=(float(cfg.min_score_max) if cfg.min_score_max is not None else None),
            min_score_mean=(
                float(cfg.min_score_mean) if cfg.min_score_mean is not None else None
            ),
            merge_nearby_enabled=bool(cfg.merge_nearby_enabled),
            merge_nearby_max_gap_px=int(cfg.merge_nearby_max_gap_px),
            max_regions_sort_by=str(cfg.max_regions_sort_by),
            max_regions=(int(cfg.max_regions) if cfg.max_regions is not None else None),
        )
        saved_defects_mask = defects["mask"]

        mask_meta: dict[str, Any] = {
            "shape": [int(d) for d in defects["mask"].shape],
            "dtype": str(defects["mask"].dtype),
        }
        if cfg.masks_dir is not None:
            out_dir = Path(cfg.masks_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            stem = Path(request.input_path).stem
            fmt = str(cfg.mask_format)
            if fmt == "png":
                ext = ".png"
            elif fmt == "npy":
                ext = ".npy"
            elif fmt == "npz":
                ext = ".npz"
            else:  # pragma: no cover - guarded by argparse choices
                raise ValueError(f"Unknown mask_format: {fmt!r}")
            out_path = out_dir / f"{int(request.index):06d}_{stem}{ext}"
            written = save_binary_mask(defects["mask"], out_path, format=str(cfg.mask_format))
            mask_meta.update(
                {
                    "path": str(written),
                    "encoding": str(cfg.mask_format),
                }
            )
        else:
            mask_meta["encoding"] = str(cfg.mask_format)

        if bool(cfg.defects_image_space):
            try:
                _project_regions_to_image_space(
                    input_path=str(request.input_path),
                    regions=defects["regions"],
                    src_hw=(int(defects["mask"].shape[0]), int(defects["mask"].shape[1])),
                )
            except Exception:
                pass

        record["defects"] = {
            "space": defects["space"],
            "pixel_threshold": float(cfg.pixel_threshold_value),
            "pixel_threshold_provenance": dict(cfg.pixel_threshold_provenance),
            "mask": mask_meta,
            "regions": defects["regions"],
            "map_stats_roi": defects.get("map_stats_roi", None),
        }
        regions_payload = {
            "index": int(request.index),
            "input": str(request.input_path),
            "defects": {
                "space": defects["space"],
                "pixel_threshold": float(cfg.pixel_threshold_value),
                "pixel_threshold_provenance": dict(cfg.pixel_threshold_provenance),
                "regions": defects["regions"],
                "map_stats_roi": defects.get("map_stats_roi", None),
            },
        }

    _save_overlay(
        index=int(request.index),
        input_path=str(request.input_path),
        anomaly_map=anomaly_map,
        defect_mask=saved_defects_mask,
        overlays_dir=request.overlays_dir,
    )

    return InferResultArtifactResult(
        record=record,
        regions_payload=regions_payload,
    )


__all__ = [
    "DefectsArtifactConfig",
    "DefectsArtifactConfigBuildRequest",
    "InferArtifactOptions",
    "InferResultArtifactCliRequest",
    "InferResultArtifactAssemblyRequest",
    "InferResultArtifactBuildRequest",
    "InferResultArtifactRequest",
    "InferResultArtifactResult",
    "build_defects_artifact_config",
    "build_infer_result_artifact_build_request_from_cli",
    "build_infer_result_artifact_request",
    "build_infer_result_artifact_request_from_cli",
    "build_infer_result_artifact_request_from_options",
    "materialize_infer_result_artifacts",
]
