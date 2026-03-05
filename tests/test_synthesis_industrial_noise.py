from __future__ import annotations

import numpy as np


def test_industrial_noise_ops_smoke() -> None:
    from pyimgano.synthesis.industrial_noise import dust_specks, stripe_noise, vibration_blur

    img = np.full((64, 80, 3), 120, dtype=np.uint8)
    rng = np.random.default_rng(0)

    out1 = vibration_blur(img, severity=2, rng=rng)
    assert out1.shape == img.shape
    assert out1.dtype == np.uint8

    out2 = stripe_noise(img, severity=3, rng=np.random.default_rng(1), direction="horizontal")
    assert out2.shape == img.shape
    assert out2.dtype == np.uint8

    out3 = dust_specks(img, severity=4, rng=np.random.default_rng(2))
    assert out3.shape == img.shape
    assert out3.dtype == np.uint8
