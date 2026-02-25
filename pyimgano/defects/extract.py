from __future__ import annotations

from typing import Sequence

import numpy as np

from pyimgano.defects.binary_postprocess import postprocess_binary_mask
from pyimgano.defects.map_ops import apply_border_ignore_to_map, apply_roi_to_map, compute_roi_stats
from pyimgano.defects.mask import anomaly_map_to_binary_mask
from pyimgano.defects.regions import extract_regions_from_mask
from pyimgano.defects.roi import roi_mask_from_xyxy_norm
from pyimgano.defects.smoothing import smooth_anomaly_map
from pyimgano.defects.hysteresis import hysteresis_anomaly_map_to_binary_mask


def extract_defects_from_anomaly_map(
    anomaly_map: np.ndarray,
    *,
    pixel_threshold: float,
    roi_xyxy_norm: Sequence[float] | None,
    border_ignore_px: int = 0,
    map_smoothing_method: str = "none",
    map_smoothing_ksize: int = 0,
    map_smoothing_sigma: float = 0.0,
    hysteresis_enabled: bool = False,
    hysteresis_low: float | None = None,
    hysteresis_high: float | None = None,
    open_ksize: int,
    close_ksize: int,
    fill_holes: bool,
    min_area: int,
    min_fill_ratio: float | None = None,
    max_aspect_ratio: float | None = None,
    min_solidity: float | None = None,
    min_score_max: float | None = None,
    min_score_mean: float | None = None,
    merge_nearby_enabled: bool = False,
    merge_nearby_max_gap_px: int = 0,
    max_regions_sort_by: str = "score_max",
    max_regions: int | None,
) -> dict:
    """Extract industrial "defects" from an anomaly map.

    Output coordinates are in anomaly-map pixel space.
    """

    amap = np.asarray(anomaly_map, dtype=np.float32)
    if amap.ndim != 2:
        raise ValueError(f"anomaly_map must be 2D (H, W), got shape {amap.shape}")

    amap_roi = apply_roi_to_map(amap, roi_xyxy_norm=roi_xyxy_norm)
    amap_roi = apply_border_ignore_to_map(amap_roi, border_ignore_px=int(border_ignore_px))
    amap_roi = smooth_anomaly_map(
        amap_roi,
        method=str(map_smoothing_method),
        ksize=int(map_smoothing_ksize),
        sigma=float(map_smoothing_sigma),
    )
    map_stats_roi = compute_roi_stats(amap, roi_xyxy_norm=roi_xyxy_norm)

    if bool(hysteresis_enabled):
        high = float(hysteresis_high) if hysteresis_high is not None else float(pixel_threshold)
        low = float(hysteresis_low) if hysteresis_low is not None else float(high) * 0.5
        mask = hysteresis_anomaly_map_to_binary_mask(amap_roi, low=float(low), high=float(high))
    else:
        mask = anomaly_map_to_binary_mask(amap_roi, pixel_threshold=float(pixel_threshold))
    mask = postprocess_binary_mask(
        mask,
        min_area=int(min_area),
        open_ksize=int(open_ksize),
        close_ksize=int(close_ksize),
        fill_holes=bool(fill_holes),
        anomaly_map=amap_roi,
        min_score_max=(float(min_score_max) if min_score_max is not None else None),
        min_score_mean=(float(min_score_mean) if min_score_mean is not None else None),
        min_fill_ratio=(float(min_fill_ratio) if min_fill_ratio is not None else None),
        max_aspect_ratio=(float(max_aspect_ratio) if max_aspect_ratio is not None else None),
        min_solidity=(float(min_solidity) if min_solidity is not None else None),
    )

    if roi_xyxy_norm is not None:
        roi_mask = roi_mask_from_xyxy_norm(amap.shape, roi_xyxy_norm)
        mask = (mask > 0).astype(np.uint8) * 255
        mask = mask * roi_mask

    include_shape_stats = any(
        v is not None for v in (min_fill_ratio, max_aspect_ratio, min_solidity)
    )
    regions = extract_regions_from_mask(
        mask,
        anomaly_map=amap_roi,
        include_shape_stats=bool(include_shape_stats),
        include_solidity=(min_solidity is not None),
    )
    if bool(merge_nearby_enabled) and int(merge_nearby_max_gap_px) > 0 and len(regions) > 1:
        from pyimgano.defects.merge import merge_regions_nearby

        regions = merge_regions_nearby(regions, max_gap_px=int(merge_nearby_max_gap_px))
    sort_by = str(max_regions_sort_by or "score_max").lower().strip()
    if sort_by not in ("score_max", "score_mean", "area"):
        raise ValueError(f"max_regions_sort_by must be one of: score_max|score_mean|area, got {sort_by!r}")

    def _primary(r: dict) -> float:
        if sort_by == "score_mean":
            return float(r.get("score_mean", 0.0) or 0.0)
        if sort_by == "area":
            return float(r.get("area", 0) or 0)
        return float(r.get("score_max", 0.0) or 0.0)

    def _sort_key(r: dict) -> tuple:
        bbox = r.get("bbox_xyxy") or [0, 0, 0, 0]
        centroid = r.get("centroid_xy") or [0.0, 0.0]
        return (
            -_primary(r),
            -float(r.get("score_max", 0.0) or 0.0),
            -float(r.get("score_mean", 0.0) or 0.0),
            -int(r.get("area", 0) or 0),
            int(bbox[1]),
            int(bbox[0]),
            int(bbox[3]),
            int(bbox[2]),
            float(centroid[1]),
            float(centroid[0]),
            int(r.get("id", 0) or 0),
        )

    regions = sorted(regions, key=_sort_key)
    if max_regions is not None:
        regions = regions[: int(max_regions)]

    return {
        "space": {"type": "anomaly_map", "shape": [int(amap.shape[0]), int(amap.shape[1])]},
        "pixel_threshold": float(pixel_threshold),
        "mask": mask,
        "regions": regions,
        "map_stats_roi": map_stats_roi,
    }
