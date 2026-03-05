from __future__ import annotations

import numpy as np


def test_guided_filter_preserves_shape_and_dtype() -> None:
    from pyimgano.preprocessing.guided_filter import guided_filter

    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(64, 64), dtype=np.uint8)
    out = guided_filter(img, radius=4, eps=1e-3)

    assert out.shape == img.shape
    assert out.dtype == np.uint8
    assert np.isfinite(out.astype(np.float32)).all()


def test_guided_filter_constant_image_is_fixed_point() -> None:
    from pyimgano.preprocessing.guided_filter import guided_filter

    img = np.ones((32, 32), dtype=np.uint8) * 128
    out = guided_filter(img, radius=8, eps=1e-3)

    assert out.shape == img.shape
    assert out.dtype == np.uint8
    assert int(out.min()) == 128
    assert int(out.max()) == 128
