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


def test_random_brush_stroke_mask_smoke() -> None:
    from pyimgano.synthesis.masks import random_brush_stroke_mask

    rng = np.random.default_rng(3)
    m = random_brush_stroke_mask((64, 64), rng=rng, num_strokes=3, thickness_range=(6, 14))
    _assert_u8_binary(m, shape_hw=(64, 64))
    assert int(np.sum(m > 0)) > 0


def test_random_spatter_mask_smoke() -> None:
    from pyimgano.synthesis.masks import random_spatter_mask

    rng = np.random.default_rng(4)
    m = random_spatter_mask((80, 64), rng=rng, num_droplets=40, radius_range=(1, 4), blur_sigma=1.0)
    _assert_u8_binary(m, shape_hw=(80, 64))
    assert int(np.sum(m > 0)) > 0


def test_random_edge_band_mask_smoke() -> None:
    from pyimgano.synthesis.masks import random_edge_band_mask

    rng = np.random.default_rng(5)
    m = random_edge_band_mask((48, 96), rng=rng, width_fraction_range=(0.05, 0.15))
    _assert_u8_binary(m, shape_hw=(48, 96))
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


def test_new_masks_can_be_constrained_by_roi_mask() -> None:
    """ROI constraints are applied at synthesis time via apply_roi_mask().

    This test targets the newer "industrial" masks to ensure they still behave
    correctly under ROI cropping.
    """

    from pyimgano.synthesis.masks import (
        apply_roi_mask,
        random_brush_stroke_mask,
        random_edge_band_mask,
        random_spatter_mask,
    )

    rng = np.random.default_rng(123)
    shape = (64, 64)

    roi = np.zeros(shape, dtype=np.uint8)
    roi[16:48, 16:48] = 255  # central square ROI

    m1 = random_brush_stroke_mask(shape, rng=rng, num_strokes=2, thickness_range=(6, 12))
    o1 = apply_roi_mask(m1, roi)
    _assert_u8_binary(o1, shape_hw=shape)
    assert int(np.sum(o1[(roi == 0)] > 0)) == 0

    m2 = random_spatter_mask(shape, rng=rng, num_droplets=64, radius_range=(1, 3), blur_sigma=0.8)
    o2 = apply_roi_mask(m2, roi)
    _assert_u8_binary(o2, shape_hw=shape)
    assert int(np.sum(o2[(roi == 0)] > 0)) == 0

    # Edge-band anomalies may become empty after ROI cropping (expected).
    m3 = random_edge_band_mask(shape, rng=rng, width_fraction_range=(0.08, 0.12))
    o3 = apply_roi_mask(m3, roi)
    _assert_u8_binary(o3, shape_hw=shape)
    assert int(np.sum(o3[(roi == 0)] > 0)) == 0
