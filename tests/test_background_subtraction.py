from __future__ import annotations

import numpy as np


def test_rolling_ball_background_subtraction_highlights_defect() -> None:
    from pyimgano.preprocessing.background import subtract_background_rolling_ball

    img = np.ones((64, 64), dtype=np.uint8) * 50
    img[28:36, 28:36] = 200  # bright defect on flat background

    out = subtract_background_rolling_ball(img, radius=15)

    assert out.shape == img.shape
    assert out.dtype == np.uint8
    # Background mostly removed.
    assert int(np.median(out)) == 0
    # Defect should remain visible.
    assert int(out[32, 32]) > 50


def test_rolling_ball_background_subtraction_constant_image_is_zero() -> None:
    from pyimgano.preprocessing.background import subtract_background_rolling_ball

    img = np.ones((32, 32), dtype=np.uint8) * 123
    out = subtract_background_rolling_ball(img, radius=10)
    assert out.shape == img.shape
    assert out.dtype == np.uint8
    assert int(out.min()) == 0
    assert int(out.max()) == 0
