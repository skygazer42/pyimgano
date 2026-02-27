from __future__ import annotations

import numpy as np


def _assert_u8_binary(mask: np.ndarray, *, shape_hw: tuple[int, int]) -> None:
    assert mask.shape == shape_hw
    assert mask.dtype == np.uint8
    uniq = set(np.unique(mask).tolist())
    assert uniq.issubset({0, 255})


def test_random_blob_mask_smoke() -> None:
    from pyimgano.synthesis.masks import random_blob_mask

    rng = np.random.default_rng(0)
    m = random_blob_mask((64, 80), rng=rng, num_blobs=3, radius_range=(5, 10), blur_sigma=1.5)
    _assert_u8_binary(m, shape_hw=(64, 80))
    assert int(np.sum(m > 0)) > 0


def test_random_ellipse_mask_smoke() -> None:
    from pyimgano.synthesis.masks import random_ellipse_mask

    rng = np.random.default_rng(1)
    m = random_ellipse_mask((32, 32), rng=rng, num_ellipses=2, axis_range=(3, 8), blur_sigma=1.0)
    _assert_u8_binary(m, shape_hw=(32, 32))
    assert int(np.sum(m > 0)) > 0


def test_random_scratch_mask_smoke() -> None:
    from pyimgano.synthesis.masks import random_scratch_mask

    rng = np.random.default_rng(2)
    m = random_scratch_mask((48, 64), rng=rng, num_scratches=2)
    _assert_u8_binary(m, shape_hw=(48, 64))
    assert int(np.sum(m > 0)) > 0


def test_apply_roi_mask_keeps_inside_roi() -> None:
    from pyimgano.synthesis.masks import apply_roi_mask

    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 2:6] = 255

    roi = np.zeros((8, 8), dtype=np.uint8)
    roi[4:, :] = 255

    out = apply_roi_mask(mask, roi)
    _assert_u8_binary(out, shape_hw=(8, 8))
    # Only bottom half of the original square remains.
    assert int(np.sum(out[0:4, :] > 0)) == 0
    assert int(np.sum(out[4:, :] > 0)) > 0

