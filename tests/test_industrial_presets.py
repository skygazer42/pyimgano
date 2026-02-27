from __future__ import annotations

import numpy as np


def test_shading_correction_preserves_shape_and_dtype_for_grayscale() -> None:
    from pyimgano.preprocessing.industrial_presets import shading_correction

    img = np.ones((32, 32), dtype=np.uint8) * 123
    out = shading_correction(img, radius=10, clahe_clip_limit=2.0, clahe_tile_grid_size=(4, 4))

    assert out.shape == img.shape
    assert out.dtype == np.uint8
    # Shading correction should keep a constant image constant (value may shift due to CLAHE).
    assert int(out.min()) == int(out.max())


def test_shading_correction_preserves_shape_and_dtype_for_color() -> None:
    from pyimgano.preprocessing.industrial_presets import shading_correction

    img = np.ones((32, 32, 3), dtype=np.uint8) * 50
    img[..., 1] = 120
    img[..., 2] = 200
    out = shading_correction(img, radius=10, clahe_clip_limit=2.0, clahe_tile_grid_size=(4, 4))

    assert out.shape == img.shape
    assert out.dtype == np.uint8
    assert np.isfinite(out.astype(np.float32)).all()


def test_defect_amplification_constant_image_is_zero() -> None:
    from pyimgano.preprocessing.industrial_presets import defect_amplification

    img = np.ones((32, 32), dtype=np.uint8) * 123
    out = defect_amplification(img, tophat_ksize=(9, 9), edge_method="sobel")
    assert out.shape == img.shape
    assert out.dtype == np.uint8
    assert int(out.min()) == 0
    assert int(out.max()) == 0


def test_defect_amplification_highlights_small_bright_defect() -> None:
    from pyimgano.preprocessing.industrial_presets import defect_amplification

    img = np.ones((64, 64), dtype=np.uint8) * 50
    img[30:34, 30:34] = 200
    out = defect_amplification(img, tophat_ksize=(15, 15), edge_method="sobel")
    assert out.shape == img.shape
    assert out.dtype == np.uint8
    assert int(out.max()) > 0


def test_retinex_illumination_normalization_smoke() -> None:
    from pyimgano.preprocessing.industrial_presets import retinex_illumination_normalization

    img = np.ones((32, 32, 3), dtype=np.uint8) * 50
    img[..., 1] = 120
    img[..., 2] = 200
    out = retinex_illumination_normalization(img, sigmas=(5.0, 15.0), clip_percentiles=(2.0, 98.0))

    assert out.shape == img.shape
    assert out.dtype == np.uint8
    assert np.isfinite(out.astype(np.float32)).all()
