import numpy as np

from pyimgano.defects.extract import extract_defects_from_anomaly_map


def test_extract_defects_from_anomaly_map_returns_mask_and_regions() -> None:
    amap = np.zeros((8, 8), dtype=np.float32)
    amap[2:5, 3:6] = 1.0
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
    assert out["mask"].shape == (8, 8)
    assert len(out["regions"]) == 1
    assert out["space"]["type"] == "anomaly_map"


def test_extract_defects_from_anomaly_map_sorts_and_limits_regions() -> None:
    amap = np.zeros((8, 8), dtype=np.float32)
    # Region id order is roughly scan order (top-left to bottom-right).
    amap[0, 0] = 0.95  # id 1, area 1
    amap[2:4, 2:4] = 0.9  # id 2, area 4
    amap[2:4, 6:8] = 0.9  # id 3, area 4 (tie with id 2)

    out = extract_defects_from_anomaly_map(
        amap,
        pixel_threshold=0.5,
        roi_xyxy_norm=None,
        open_ksize=0,
        close_ksize=0,
        fill_holes=False,
        min_area=0,
        max_regions=2,
    )
    assert len(out["regions"]) == 2
    assert [r["id"] for r in out["regions"]] == [1, 2]


def test_extract_defects_from_anomaly_map_can_filter_by_region_score() -> None:
    amap = np.zeros((8, 8), dtype=np.float32)
    amap[1:3, 1:3] = 0.6  # region A (kept)
    amap[5:7, 5:7] = 0.4  # region B (filtered out)

    out = extract_defects_from_anomaly_map(
        amap,
        pixel_threshold=0.3,
        roi_xyxy_norm=None,
        open_ksize=0,
        close_ksize=0,
        fill_holes=False,
        min_area=0,
        min_score_max=0.5,
        max_regions=None,
    )
    assert len(out["regions"]) == 1
    assert int(out["mask"][2, 2]) == 255
    assert int(out["mask"][6, 6]) == 0
