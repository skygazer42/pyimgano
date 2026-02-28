from __future__ import annotations

import numpy as np

from pyimgano.defects.extract import extract_defects_from_anomaly_map


def test_extract_defects_can_filter_regions_by_score_quantile() -> None:
    amap = np.zeros((8, 8), dtype=np.float32)
    amap[1:3, 1:3] = 0.6  # region low
    amap[5:7, 5:7] = 0.9  # region high

    out = extract_defects_from_anomaly_map(
        amap,
        pixel_threshold=0.5,
        roi_xyxy_norm=None,
        open_ksize=0,
        close_ksize=0,
        fill_holes=False,
        min_area=0,
        max_regions=None,
        min_score_max_quantile=0.5,
    )

    assert len(out["regions"]) == 1
    assert out["regions"][0]["bbox_xyxy"] == [5, 5, 6, 6]

