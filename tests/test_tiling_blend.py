from __future__ import annotations

import numpy as np


def test_tile_apply_identity_preserves_image_exactly() -> None:
    from pyimgano.preprocessing.tiling import tile_apply

    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(128, 160, 3), dtype=np.uint8)

    out = tile_apply(img, lambda x: x, tile_size=64, overlap=16, blend="hann")
    assert out.shape == img.shape
    assert out.dtype == np.uint8
    assert np.array_equal(out, img)


def test_tile_apply_handles_small_images() -> None:
    from pyimgano.preprocessing.tiling import tile_apply

    img = np.ones((40, 50), dtype=np.uint8) * 123
    out = tile_apply(img, lambda x: x, tile_size=64, overlap=16, blend="hann")
    assert out.shape == img.shape
    assert out.dtype == np.uint8
    assert np.array_equal(out, img)
