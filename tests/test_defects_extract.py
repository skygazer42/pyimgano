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

