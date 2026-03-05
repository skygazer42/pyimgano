from __future__ import annotations

import numpy as np


def test_msrcr_lite_smoke() -> None:
    from pyimgano.preprocessing.retinex import msrcr_lite

    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(64, 80, 3), dtype=np.uint8)
    out = msrcr_lite(img, sigmas=(5.0, 20.0), clip_percentiles=(2.0, 98.0))
    assert out.shape == img.shape
    assert out.dtype == np.uint8
    # Should change the image in most cases.
    assert not np.array_equal(out, img)
