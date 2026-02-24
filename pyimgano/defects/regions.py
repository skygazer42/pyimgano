from __future__ import annotations

import cv2
import numpy as np


def extract_regions_from_mask(mask_u8: np.ndarray, anomaly_map: np.ndarray | None = None) -> list[dict]:
    """Extract connected-component regions from a binary mask (0/255).

    Returns regions in anomaly-map coordinate space with:
    - bbox_xyxy: inclusive pixel bounds [x1, y1, x2, y2]
    - area: number of foreground pixels
    - centroid_xy: subpixel centroid [x, y]
    """

    if mask_u8.ndim != 2:
        raise ValueError(f"mask_u8 must be 2D (H, W), got shape {mask_u8.shape}")

    binary01 = (np.asarray(mask_u8, dtype=np.uint8) > 0).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary01, connectivity=8)

    amap: np.ndarray | None
    if anomaly_map is None:
        amap = None
    else:
        amap = np.asarray(anomaly_map, dtype=np.float32)
        if amap.shape != binary01.shape:
            raise ValueError(
                f"anomaly_map must have shape {binary01.shape}, got {amap.shape}"
            )

    regions: list[dict] = []
    for label_id in range(1, num_labels):
        left = int(stats[label_id, cv2.CC_STAT_LEFT])
        top = int(stats[label_id, cv2.CC_STAT_TOP])
        width = int(stats[label_id, cv2.CC_STAT_WIDTH])
        height = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        cx, cy = centroids[label_id]

        x1, y1 = left, top
        x2, y2 = left + width - 1, top + height - 1

        regions.append(
            {
                "id": int(label_id),
                "bbox_xyxy": [x1, y1, x2, y2],
                "area": int(area),
                "centroid_xy": [float(cx), float(cy)],
            }
        )

        if amap is not None:
            region_values = amap[labels == label_id]
            if region_values.size:
                regions[-1]["score_max"] = round(float(region_values.max()), 6)
                regions[-1]["score_mean"] = round(float(region_values.mean()), 6)

    return regions
