import numpy as np

from pyimgano.defects.binary_postprocess import postprocess_binary_mask


def test_postprocess_binary_mask_removes_small_components() -> None:
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[1, 1] = 255  # tiny
    mask[5:8, 5:8] = 255  # big
    out = postprocess_binary_mask(
        mask,
        min_area=4,
        open_ksize=0,
        close_ksize=0,
        fill_holes=False,
    )
    assert int(out[1, 1]) == 0
    assert int(out[6, 6]) == 255


def test_postprocess_binary_mask_fill_holes_fills_internal_hole() -> None:
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:8, 2:8] = 255
    mask[4:6, 4:6] = 0  # hole
    out = postprocess_binary_mask(
        mask,
        min_area=0,
        open_ksize=0,
        close_ksize=0,
        fill_holes=True,
    )
    assert int(out[5, 5]) == 255


def test_postprocess_binary_mask_can_filter_components_by_score_max() -> None:
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[0, 0] = 255
    mask[3, 3] = 255

    amap = np.zeros((4, 4), dtype=np.float32)
    amap[0, 0] = 0.2
    amap[3, 3] = 0.9

    out = postprocess_binary_mask(
        mask,
        min_area=0,
        open_ksize=0,
        close_ksize=0,
        fill_holes=False,
        anomaly_map=amap,
        min_score_max=0.5,
    )
    assert int(out[0, 0]) == 0
    assert int(out[3, 3]) == 255


def test_postprocess_binary_mask_min_score_filters_require_anomaly_map() -> None:
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[0, 0] = 255

    try:
        postprocess_binary_mask(
            mask,
            min_area=0,
            open_ksize=0,
            close_ksize=0,
            fill_holes=False,
            min_score_max=0.5,
        )
        raised = False
    except ValueError:
        raised = True

    assert raised is True
