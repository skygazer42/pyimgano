from __future__ import annotations

import numpy as np

from pyimgano.io.image import read_image, resize_image


def test_read_image_bgr_rgb_and_resize(tmp_path) -> None:
    import cv2

    bgr = np.zeros((2, 2, 3), dtype=np.uint8)
    bgr[0, 0] = np.asarray([10, 20, 30], dtype=np.uint8)  # B,G,R

    path = tmp_path / "x.png"
    assert cv2.imwrite(str(path), bgr) is True

    loaded_bgr = read_image(path, color="bgr")
    assert loaded_bgr.shape == (2, 2, 3)
    assert loaded_bgr[0, 0].tolist() == [10, 20, 30]

    loaded_rgb = read_image(path, color="rgb")
    assert loaded_rgb.shape == (2, 2, 3)
    assert loaded_rgb[0, 0].tolist() == [30, 20, 10]

    resized = resize_image(loaded_bgr, (1, 1))
    assert resized.shape == (1, 1, 3)


def test_read_image_missing_path_raises(tmp_path) -> None:
    missing = tmp_path / "missing.png"
    try:
        read_image(missing, color="bgr")
    except FileNotFoundError:
        return
    raise AssertionError("Expected FileNotFoundError")

