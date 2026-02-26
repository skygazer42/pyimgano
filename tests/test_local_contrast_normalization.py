from __future__ import annotations

import numpy as np


def test_local_contrast_normalization_runs_and_preserves_shape() -> None:
    from pyimgano.preprocessing.enhancer import ImageEnhancer

    enh = ImageEnhancer()
    img = np.zeros((64, 64), dtype=np.uint8)
    img[16:48, 16:48] = 200

    out = enh.local_contrast_normalization(img, ksize=15, clip=3.0)
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_local_contrast_normalization_constant_image_maps_to_midgray() -> None:
    from pyimgano.preprocessing.enhancer import ImageEnhancer

    enh = ImageEnhancer()
    img = np.ones((32, 32), dtype=np.uint8) * 123
    out = enh.local_contrast_normalization(img, ksize=9, clip=3.0)
    assert out.shape == img.shape
    assert out.dtype == np.uint8
    # Zero-contrast input should map to the neutral center.
    assert int(out.min()) == 128
    assert int(out.max()) == 128

