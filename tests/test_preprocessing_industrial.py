from __future__ import annotations

import numpy as np

from pyimgano.preprocessing.industrial_presets import (
    gray_world_white_balance,
    homomorphic_filter,
    max_rgb_white_balance,
)


def test_white_balance_preserves_shape_and_dtype() -> None:
    img = np.random.randint(0, 255, size=(32, 32, 3), dtype=np.uint8)
    out1 = gray_world_white_balance(img)
    out2 = max_rgb_white_balance(img)

    assert out1.shape == img.shape
    assert out2.shape == img.shape
    assert out1.dtype == np.uint8
    assert out2.dtype == np.uint8


def test_white_balance_noop_for_grayscale() -> None:
    img = np.random.randint(0, 255, size=(16, 16), dtype=np.uint8)
    out1 = gray_world_white_balance(img)
    out2 = max_rgb_white_balance(img)
    assert np.array_equal(out1, img)
    assert np.array_equal(out2, img)


def test_homomorphic_filter_runs_and_preserves_contract() -> None:
    img = np.random.randint(0, 255, size=(32, 32, 3), dtype=np.uint8)
    out = homomorphic_filter(img, cutoff=0.6, gamma_low=0.7, gamma_high=1.3)
    assert out.shape == img.shape
    assert out.dtype == np.uint8
    assert out.min() >= 0
    assert out.max() <= 255

