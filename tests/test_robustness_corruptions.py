from __future__ import annotations

import numpy as np

from pyimgano.robustness.corruptions import (
    apply_blur,
    apply_geo_jitter,
    apply_glare,
    apply_jpeg,
    apply_lighting,
)


def test_apply_lighting_deterministic_for_fixed_seed() -> None:
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    img[4:12, 4:12] = 128

    out1, _ = apply_lighting(img, mask=None, severity=3, rng=np.random.default_rng(0))
    out2, _ = apply_lighting(img, mask=None, severity=3, rng=np.random.default_rng(0))

    assert out1.shape == img.shape
    assert out1.dtype == np.uint8
    assert np.array_equal(out1, out2)


def test_apply_jpeg_deterministic_for_fixed_seed() -> None:
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    img[2:14, 2:14] = 200

    out1, _ = apply_jpeg(img, mask=None, severity=4, rng=np.random.default_rng(0))
    out2, _ = apply_jpeg(img, mask=None, severity=4, rng=np.random.default_rng(0))

    assert out1.shape == img.shape
    assert out1.dtype == np.uint8
    assert np.array_equal(out1, out2)


def test_apply_blur_deterministic_for_fixed_seed() -> None:
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    img[2:14, 2:14] = 255

    out1, _ = apply_blur(img, mask=None, severity=2, rng=np.random.default_rng(0))
    out2, _ = apply_blur(img, mask=None, severity=2, rng=np.random.default_rng(0))

    assert out1.shape == img.shape
    assert out1.dtype == np.uint8
    assert np.array_equal(out1, out2)


def test_apply_glare_deterministic_for_fixed_seed() -> None:
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    img[8:24, 8:24] = 120

    out1, _ = apply_glare(img, mask=None, severity=5, rng=np.random.default_rng(0))
    out2, _ = apply_glare(img, mask=None, severity=5, rng=np.random.default_rng(0))

    assert out1.shape == img.shape
    assert out1.dtype == np.uint8
    assert np.array_equal(out1, out2)


def test_apply_geo_jitter_warps_mask_deterministically() -> None:
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    img[8:24, 8:24] = 180

    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[10:22, 10:22] = 1

    out_img1, out_mask1 = apply_geo_jitter(
        img, mask=mask, severity=4, rng=np.random.default_rng(0)
    )
    out_img2, out_mask2 = apply_geo_jitter(
        img, mask=mask, severity=4, rng=np.random.default_rng(0)
    )

    assert out_img1.shape == img.shape
    assert out_img1.dtype == np.uint8
    assert out_mask1 is not None
    assert out_mask1.shape == mask.shape
    assert set(np.unique(out_mask1).tolist()).issubset({0, 1})
    assert int(out_mask1.sum()) > 0

    assert np.array_equal(out_img1, out_img2)
    assert np.array_equal(out_mask1, out_mask2)
