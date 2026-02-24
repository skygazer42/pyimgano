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

