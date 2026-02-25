from __future__ import annotations

from collections import defaultdict
from typing import Sequence


def _bbox_gap_px(a_xyxy: Sequence[int], b_xyxy: Sequence[int]) -> tuple[int, int]:
    ax1, ay1, ax2, ay2 = (int(v) for v in a_xyxy)
    bx1, by1, bx2, by2 = (int(v) for v in b_xyxy)

    # Inclusive bboxes: a gap of 0 means touching edges (ax2 == bx1 - 1).
    if ax2 < bx1:
        gap_x = bx1 - ax2 - 1
    elif bx2 < ax1:
        gap_x = ax1 - bx2 - 1
    else:
        gap_x = 0

    if ay2 < by1:
        gap_y = by1 - ay2 - 1
    elif by2 < ay1:
        gap_y = ay1 - by2 - 1
    else:
        gap_y = 0

    return (int(gap_x), int(gap_y))


def _regions_within_gap(a: dict, b: dict, *, max_gap_px: int) -> bool:
    if max_gap_px <= 0:
        return False
    a_bbox = a.get("bbox_xyxy", None)
    b_bbox = b.get("bbox_xyxy", None)
    if a_bbox is None or b_bbox is None:
        return False

    gap_x, gap_y = _bbox_gap_px(a_bbox, b_bbox)
    return int(gap_x) <= int(max_gap_px) and int(gap_y) <= int(max_gap_px)


def merge_regions_nearby(regions: Sequence[dict], *, max_gap_px: int) -> list[dict]:
    """Best-effort merging of nearby regions based on bbox gap.

    This is intended for industrial defects outputs where a single physical defect
    can appear as multiple fragments (e.g. due to threshold speckle, seam cuts).

    Notes:
    - Does not modify masks; only merges the *regions list*.
    - Merge condition: bbox gap in both x/y is <= max_gap_px.
    - Runs in O(N^2) over region count; regions are expected to be small (topK).
    """

    n = int(len(regions))
    if n <= 1:
        return list(regions)

    gap = int(max_gap_px)
    if gap <= 0:
        return list(regions)

    # Union-find over region indices (deterministic: always attach higher root to lower root).
    parent = list(range(n))

    def _find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def _union(i: int, j: int) -> None:
        ri, rj = _find(i), _find(j)
        if ri == rj:
            return
        if ri < rj:
            parent[rj] = ri
        else:
            parent[ri] = rj

    for i in range(n):
        a = regions[i]
        for j in range(i + 1, n):
            b = regions[j]
            if _regions_within_gap(a, b, max_gap_px=gap):
                _union(i, j)

    clusters: dict[int, list[int]] = defaultdict(list)
    for idx in range(n):
        clusters[_find(idx)].append(idx)

    merged: list[dict] = []
    for _root, idxs in clusters.items():
        if len(idxs) == 1:
            merged.append(dict(regions[idxs[0]]))
            continue

        members = [regions[i] for i in idxs]

        member_ids: list[int] = []
        for m in members:
            try:
                member_ids.append(int(m.get("id")))
            except Exception:
                continue
        member_ids = sorted(set(member_ids))

        xs1: list[int] = []
        ys1: list[int] = []
        xs2: list[int] = []
        ys2: list[int] = []
        areas: list[int] = []
        cx_sum = 0.0
        cy_sum = 0.0
        score_max: float | None = None
        score_mean_num = 0.0
        score_mean_den = 0.0

        for m in members:
            bbox = m.get("bbox_xyxy", None)
            if bbox is not None and len(bbox) == 4:
                x1, y1, x2, y2 = (int(v) for v in bbox)
                xs1.append(x1)
                ys1.append(y1)
                xs2.append(x2)
                ys2.append(y2)

            area = int(m.get("area", 0) or 0)
            areas.append(area)

            centroid = m.get("centroid_xy", None)
            if centroid is not None and len(centroid) == 2 and area > 0:
                cx_sum += float(centroid[0]) * float(area)
                cy_sum += float(centroid[1]) * float(area)

            v_max = m.get("score_max", None)
            if v_max is not None:
                try:
                    score_max = float(v_max) if score_max is None else max(score_max, float(v_max))
                except Exception:
                    pass

            v_mean = m.get("score_mean", None)
            if v_mean is not None and area > 0:
                try:
                    score_mean_num += float(v_mean) * float(area)
                    score_mean_den += float(area)
                except Exception:
                    pass

        x1m = min(xs1) if xs1 else 0
        y1m = min(ys1) if ys1 else 0
        x2m = max(xs2) if xs2 else -1
        y2m = max(ys2) if ys2 else -1

        area_total = int(sum(areas))
        if area_total > 0:
            cx = cx_sum / float(area_total)
            cy = cy_sum / float(area_total)
        else:
            cx = float(x1m + x2m) * 0.5
            cy = float(y1m + y2m) * 0.5

        width = int(x2m - x1m + 1)
        height = int(y2m - y1m + 1)
        bbox_area = int(width * height) if width > 0 and height > 0 else 0
        fill_ratio = float(area_total) / float(bbox_area) if bbox_area > 0 else 0.0

        aspect_ratio = float("inf")
        if width > 0 and height > 0:
            aspect_ratio = max(float(width) / float(height), float(height) / float(width))

        out = {
            "id": (min(member_ids) if member_ids else 1),
            "bbox_xyxy": [int(x1m), int(y1m), int(x2m), int(y2m)],
            "area": int(area_total),
            "centroid_xy": [float(cx), float(cy)],
            "bbox_area": int(bbox_area),
            "fill_ratio": round(float(fill_ratio), 6),
            "aspect_ratio": round(float(aspect_ratio), 6),
            "merged_from_ids": member_ids,
        }

        if score_max is not None:
            out["score_max"] = round(float(score_max), 6)
        if score_mean_den > 0.0:
            out["score_mean"] = round(float(score_mean_num / score_mean_den), 6)

        merged.append(out)

    return merged

