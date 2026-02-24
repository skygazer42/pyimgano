from __future__ import annotations

from typing import Sequence

import numpy as np

from pyimgano.defects.binary_postprocess import postprocess_binary_mask
from pyimgano.defects.map_ops import apply_roi_to_map, compute_roi_stats
from pyimgano.defects.mask import anomaly_map_to_binary_mask
from pyimgano.defects.regions import extract_regions_from_mask
from pyimgano.defects.roi import roi_mask_from_xyxy_norm


def extract_defects_from_anomaly_map(
    anomaly_map: np.ndarray,
    *,
    pixel_threshold: float,
    roi_xyxy_norm: Sequence[float] | None,
    open_ksize: int,
    close_ksize: int,
    fill_holes: bool,
    min_area: int,
    max_regions: int | None,
) -> dict:
    """Extract industrial "defects" from an anomaly map.

    Output coordinates are in anomaly-map pixel space.
    """

    amap = np.asarray(anomaly_map, dtype=np.float32)
    if amap.ndim != 2:
        raise ValueError(f"anomaly_map must be 2D (H, W), got shape {amap.shape}")

    amap_roi = apply_roi_to_map(amap, roi_xyxy_norm=roi_xyxy_norm)
    map_stats_roi = compute_roi_stats(amap, roi_xyxy_norm=roi_xyxy_norm)

    mask = anomaly_map_to_binary_mask(amap_roi, pixel_threshold=float(pixel_threshold))
    mask = postprocess_binary_mask(
        mask,
        min_area=int(min_area),
        open_ksize=int(open_ksize),
        close_ksize=int(close_ksize),
        fill_holes=bool(fill_holes),
    )

    if roi_xyxy_norm is not None:
        roi_mask = roi_mask_from_xyxy_norm(amap.shape, roi_xyxy_norm)
        mask = (mask > 0).astype(np.uint8) * 255
        mask = mask * roi_mask

    regions = extract_regions_from_mask(mask, anomaly_map=amap_roi)
    if max_regions is not None:
        k = int(max_regions)
        regions = sorted(
            regions,
            key=lambda r: (float(r.get("score_max", 0.0)), int(r.get("area", 0))),
            reverse=True,
        )[:k]

    return {
        "space": {"type": "anomaly_map", "shape": [int(amap.shape[0]), int(amap.shape[1])]},
        "pixel_threshold": float(pixel_threshold),
        "mask": mask,
        "regions": regions,
        "map_stats_roi": map_stats_roi,
    }

