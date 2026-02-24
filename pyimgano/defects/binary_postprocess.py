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
) -> np.ndarray:
    """Postprocess a binary defect mask (uint8, 0/255).

    Operations are applied in this order:
    1) morphology open (optional)
    2) morphology close (optional)
    3) min-area filter via connected components (optional)
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

    if min_area and min_area > 0:
        binary01 = (mask > 0).astype(np.uint8)
        num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary01, connectivity=8)
        keep = np.zeros_like(mask, dtype=np.uint8)
        for label_id in range(1, num_labels):
            area = int(stats[label_id, cv2.CC_STAT_AREA])
            if area >= int(min_area):
                keep[labels == label_id] = 255
        mask = keep

    if fill_holes:
        # Filled in Task 5; keep behavior best-effort and dependency-light.
        mask = _fill_holes_best_effort(mask)

    return mask


def _fill_holes_best_effort(mask_u8: np.ndarray) -> np.ndarray:
    # Placeholder for Task 5: dependency-light hole filling.
    return mask_u8

