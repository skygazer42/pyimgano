from __future__ import annotations

import numpy as np

from pyimgano.defects.extract import extract_defects_from_anomaly_map


def test_extract_defects_can_dilate_mask() -> None:
    amap = np.zeros((9, 9), dtype=np.float32)
    amap[4, 4] = 1.0

    out_no = extract_defects_from_anomaly_map(
        amap,
        pixel_threshold=0.5,
        roi_xyxy_norm=None,
        open_ksize=0,
        close_ksize=0,
        fill_holes=False,
        min_area=0,
        max_regions=None,
        mask_dilate_ksize=0,
    )
    out_yes = extract_defects_from_anomaly_map(
        amap,
        pixel_threshold=0.5,
        roi_xyxy_norm=None,
        open_ksize=0,
        close_ksize=0,
        fill_holes=False,
        min_area=0,
        max_regions=None,
        mask_dilate_ksize=3,
    )

    assert int(np.sum(out_no["mask"] > 0)) == 1
    assert int(np.sum(out_yes["mask"] > 0)) > 1
