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

