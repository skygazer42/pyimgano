from __future__ import annotations

import numpy as np

from pyimgano.defects.extract import extract_defects_from_anomaly_map


def test_extract_defects_includes_region_shape_stats_by_default() -> None:
    amap = np.zeros((10, 10), dtype=np.float32)
    amap[2:5, 3:7] = 1.0  # width=4, height=3, area=12

    out = extract_defects_from_anomaly_map(
        amap,
        pixel_threshold=0.5,
        roi_xyxy_norm=None,
        open_ksize=0,
        close_ksize=0,
        fill_holes=False,
        min_area=0,
        max_regions=None,
    )
    assert len(out["regions"]) == 1
    r = out["regions"][0]

    assert r["bbox_area"] == 12
    assert r["fill_ratio"] == 1.0
    assert abs(float(r["aspect_ratio"]) - (4.0 / 3.0)) < 1e-6
    assert r["solidity"] is not None
    assert 0.9 <= float(r["solidity"]) <= 1.0
