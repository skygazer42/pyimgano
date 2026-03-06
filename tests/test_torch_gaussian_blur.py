from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")


def test_torch_gaussian_blur_constant_image_is_fixed_point() -> None:
    from pyimgano.preprocessing.enhancer import ImageEnhancer

    enh = ImageEnhancer()
    img = np.ones((32, 32, 3), dtype=np.uint8) * 123
    out = enh.gaussian_blur_torch(img, kernel_size=9, sigma=2.0, device="cpu")

    assert out.shape == img.shape
    assert out.dtype == np.uint8
    assert int(out.min()) == 123
    assert int(out.max()) == 123


def test_torch_gaussian_blur_preserves_shape_and_dtype() -> None:
    from pyimgano.preprocessing.enhancer import ImageEnhancer

    rng = np.random.default_rng(0)
    enh = ImageEnhancer()
    img = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    out = enh.gaussian_blur_torch(img, kernel_size=5, sigma=1.0, device="cpu")

    assert out.shape == img.shape
    assert out.dtype == np.uint8
    assert np.isfinite(out.astype(np.float32)).all()
