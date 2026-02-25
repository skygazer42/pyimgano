from __future__ import annotations

import cv2
import numpy as np


def postprocess_binary_mask(
    mask_u8: np.ndarray,
    *,
    min_area: int,
    open_ksize: int,
    close_ksize: int,
    fill_holes: bool,
    anomaly_map: np.ndarray | None = None,
    min_score_max: float | None = None,
    min_score_mean: float | None = None,
) -> np.ndarray:
    """Postprocess a binary defect mask (uint8, 0/255).

    Operations are applied in this order:
    1) morphology open (optional)
    2) morphology close (optional)
    3) connected-component filtering (optional; min area / min score)
    4) hole filling (optional; may be a no-op if not available)
    """

    if mask_u8.ndim != 2:
        raise ValueError(f"mask_u8 must be 2D (H, W), got shape {mask_u8.shape}")

    mask = (np.asarray(mask_u8, dtype=np.uint8) > 0).astype(np.uint8) * 255

    if open_ksize and open_ksize > 0:
        k = int(open_ksize)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    if close_ksize and close_ksize > 0:
        k = int(close_ksize)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    if (min_score_max is not None or min_score_mean is not None) and anomaly_map is None:
        raise ValueError("min_score_* filters require anomaly_map to be provided.")

    should_filter_components = (
        (min_area is not None and int(min_area) > 0)
        or min_score_max is not None
        or min_score_mean is not None
    )
    if should_filter_components:
        binary01 = (mask > 0).astype(np.uint8)
        num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary01, connectivity=8)
        keep = np.zeros_like(mask, dtype=np.uint8)

        amap = None
        if anomaly_map is not None:
            amap = np.asarray(anomaly_map, dtype=np.float32)
            if amap.shape != binary01.shape:
                raise ValueError(
                    "anomaly_map must have the same shape as mask_u8 for score filtering. "
                    f"Got anomaly_map={amap.shape} vs mask_u8={binary01.shape}."
                )

        min_score_max_v = float(min_score_max) if min_score_max is not None else None
        min_score_mean_v = float(min_score_mean) if min_score_mean is not None else None

        for label_id in range(1, num_labels):
            area = int(stats[label_id, cv2.CC_STAT_AREA])
            if min_area and int(min_area) > 0 and area < int(min_area):
                continue

            if amap is not None and (min_score_max_v is not None or min_score_mean_v is not None):
                region_values = amap[labels == label_id]
                if region_values.size == 0:
                    continue
                if min_score_max_v is not None and float(region_values.max()) < float(min_score_max_v):
                    continue
                if min_score_mean_v is not None and float(region_values.mean()) < float(min_score_mean_v):
                    continue

            keep[labels == label_id] = 255
        mask = keep

    if fill_holes:
        # Filled in Task 5; keep behavior best-effort and dependency-light.
        mask = _fill_holes_best_effort(mask)

    return mask


def _fill_holes_best_effort(mask_u8: np.ndarray) -> np.ndarray:
    binary = (np.asarray(mask_u8, dtype=np.uint8) > 0)

    try:
        from scipy.ndimage import binary_fill_holes  # type: ignore[import-not-found]

        filled = binary_fill_holes(binary)
        return filled.astype(np.uint8) * 255
    except Exception:
        return _fill_holes_floodfill(mask_u8)


def _fill_holes_floodfill(mask_u8: np.ndarray) -> np.ndarray:
    """Fill internal holes using only OpenCV flood fill (no SciPy required)."""

    mask = (np.asarray(mask_u8, dtype=np.uint8) > 0).astype(np.uint8) * 255
    h, w = mask.shape
    if h == 0 or w == 0:
        return mask

    inv = cv2.bitwise_not(mask)
    flood = inv.copy()

    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

    def _flood_from(x: int, y: int) -> None:
        if flood[y, x] != 255:
            return
        flood_mask.fill(0)
        cv2.floodFill(flood, flood_mask, (x, y), 0)

    # Flood-fill all border-connected background components.
    for x in range(w):
        _flood_from(x, 0)
        _flood_from(x, h - 1)
    for y in range(h):
        _flood_from(0, y)
        _flood_from(w - 1, y)

    holes = flood
    return cv2.bitwise_or(mask, holes)
