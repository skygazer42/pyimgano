import numpy as np

from pyimgano.defects.regions import extract_regions_from_mask


def test_extract_regions_from_mask_finds_bbox_and_area() -> None:
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:5, 3:7] = 255
    regions = extract_regions_from_mask(mask)
    assert len(regions) == 1
    r = regions[0]
    assert r["bbox_xyxy"] == [3, 2, 6, 4]
    assert r["area"] == int(3 * 4)


def test_extract_regions_from_mask_adds_scores_when_map_provided() -> None:
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1:3, 1:3] = 255
    amap = np.zeros((4, 4), dtype=np.float32)
    amap[2, 2] = 0.9
    regions = extract_regions_from_mask(mask, anomaly_map=amap)
    r = regions[0]
    assert r["score_max"] == 0.9
    assert 0.0 < r["score_mean"] <= r["score_max"]


def test_extract_regions_from_mask_can_include_shape_stats() -> None:
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:5, 3:7] = 255  # width=4, height=3, area=12

    regions = extract_regions_from_mask(mask, include_shape_stats=True)
    assert len(regions) == 1
    r = regions[0]
    assert r["bbox_area"] == 12
    assert r["fill_ratio"] == 1.0
    assert abs(float(r["aspect_ratio"]) - (4.0 / 3.0)) < 1e-6


def test_extract_regions_from_mask_can_include_solidity() -> None:
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:5, 3:7] = 255  # solid rectangle (solidity ~1)

    regions = extract_regions_from_mask(mask, include_solidity=True)
    assert len(regions) == 1
    r = regions[0]
    assert r["solidity"] is not None
    assert 0.9 <= float(r["solidity"]) <= 1.0
