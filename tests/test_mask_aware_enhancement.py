from __future__ import annotations

import numpy as np


def test_mask_aware_enhancement_only_modifies_roi() -> None:
    from pyimgano.preprocessing.mixin import PreprocessingMixin

    class _D(PreprocessingMixin):
        def __init__(self):
            self.setup_preprocessing(enable=True, use_pipeline=False)

    det = _D()
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(64, 64), dtype=np.uint8)

    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[16:48, 16:48] = 1

    out = det.preprocess_image_masked(img, mask=mask, operation="gaussian_blur", ksize=(5, 5))

    assert out.shape == img.shape
    assert out.dtype == np.uint8

    # Outside ROI: unchanged
    assert np.array_equal(out[mask == 0], img[mask == 0])
    # Inside ROI: should differ for noisy image after blur
    assert not np.array_equal(out[mask == 1], img[mask == 1])

