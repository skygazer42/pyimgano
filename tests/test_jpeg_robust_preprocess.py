from __future__ import annotations

import numpy as np


def test_jpeg_robust_preprocess_preserves_shape_and_dtype() -> None:
    from pyimgano.preprocessing.enhancer import ImageEnhancer

    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    enh = ImageEnhancer()
    out = enh.jpeg_robust_preprocess(img, strength=0.7)

    assert out.shape == img.shape
    assert out.dtype == np.uint8
    assert np.isfinite(out.astype(np.float32)).all()


def test_jpeg_robust_preprocess_constant_image_is_fixed_point() -> None:
    from pyimgano.preprocessing.enhancer import ImageEnhancer

    img = np.ones((32, 32, 3), dtype=np.uint8) * 200
    enh = ImageEnhancer()
    out = enh.jpeg_robust_preprocess(img, strength=1.0)
    assert out.shape == img.shape
    assert out.dtype == np.uint8
    assert int(out.min()) == 200
    assert int(out.max()) == 200

