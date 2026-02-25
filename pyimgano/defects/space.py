from __future__ import annotations

import math
from typing import Sequence


def scale_bbox_xyxy_inclusive(
    bbox_xyxy: Sequence[int],
    *,
    src_hw: tuple[int, int],
    dst_hw: tuple[int, int],
) -> list[int]:
    """Scale an inclusive bbox from src (H,W) to dst (H,W).

    Bboxes are in `[x1, y1, x2, y2]` inclusive pixel coordinates.

    We map inclusive pixel indices by scaling pixel *edges*:
    - left/top edge: x1/y1
    - right/bottom edge (exclusive): (x2+1)/(y2+1)

    Then convert back to inclusive indices in the destination space.
    """

    x1, y1, x2, y2 = (int(v) for v in bbox_xyxy)
    src_h, src_w = (int(src_hw[0]), int(src_hw[1]))
    dst_h, dst_w = (int(dst_hw[0]), int(dst_hw[1]))

    if src_h <= 0 or src_w <= 0 or dst_h <= 0 or dst_w <= 0:
        raise ValueError(f"Invalid src/dst shapes: src={src_hw}, dst={dst_hw}")

    sx = float(dst_w) / float(src_w)
    sy = float(dst_h) / float(src_h)

    x1_edge = float(x1) * sx
    x2_edge = float(x2 + 1) * sx
    y1_edge = float(y1) * sy
    y2_edge = float(y2 + 1) * sy

    x1d = int(math.floor(x1_edge))
    y1d = int(math.floor(y1_edge))
    x2d = int(math.ceil(x2_edge) - 1)
    y2d = int(math.ceil(y2_edge) - 1)

    x1d = max(0, min(x1d, dst_w - 1))
    y1d = max(0, min(y1d, dst_h - 1))
    x2d = max(0, min(x2d, dst_w - 1))
    y2d = max(0, min(y2d, dst_h - 1))

    if x2d < x1d:
        x2d = x1d
    if y2d < y1d:
        y2d = y1d

    return [int(x1d), int(y1d), int(x2d), int(y2d)]

