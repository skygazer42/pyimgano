from __future__ import annotations

import numpy as np


def test_preprocessing_ops_preserve_shapes_dtypes_and_finiteness() -> None:
    from pyimgano.preprocessing.anisotropic_diffusion import anisotropic_diffusion
    from pyimgano.preprocessing.guided_filter import guided_filter
    from pyimgano.preprocessing.industrial_presets import defect_amplification, shading_correction
    from pyimgano.preprocessing.tiling import tile_apply
    from pyimgano.preprocessing.enhancer import ImageEnhancer

    rng = np.random.default_rng(0)
    gray = rng.integers(0, 256, size=(64, 80), dtype=np.uint8)
    color = rng.integers(0, 256, size=(64, 80, 3), dtype=np.uint8)

    # Guided filter (grayscale output)
    gf = guided_filter(color, radius=4, eps=1e-3)
    assert gf.shape == (64, 80)
    assert gf.dtype == np.uint8

    # Anisotropic diffusion (grayscale output)
    ad = anisotropic_diffusion(color, niter=3, kappa=30.0, gamma=0.1, option=1)
    assert ad.shape == (64, 80)
    assert ad.dtype == np.uint8

    # Shading correction (preserves input shape)
    sc_g = shading_correction(gray, radius=10, clahe_clip_limit=2.0, clahe_tile_grid_size=(4, 4))
    sc_c = shading_correction(color, radius=10, clahe_clip_limit=2.0, clahe_tile_grid_size=(4, 4))
    assert sc_g.shape == gray.shape
    assert sc_c.shape == color.shape
    assert sc_g.dtype == np.uint8
    assert sc_c.dtype == np.uint8

    # Defect amplification (grayscale output)
    da = defect_amplification(color, tophat_ksize=(9, 9), edge_method="sobel")
    assert da.shape == (64, 80)
    assert da.dtype == np.uint8

    # Tile+blend identity
    out = tile_apply(color, lambda x: x, tile_size=32, overlap=8, blend="hann")
    assert out.shape == color.shape
    assert out.dtype == np.uint8
    assert np.array_equal(out, color)

    enh = ImageEnhancer()
    lcn = enh.local_contrast_normalization(color, ksize=9, clip=3.0)
    assert lcn.shape == (64, 80)
    assert lcn.dtype == np.uint8

    jr = enh.jpeg_robust_preprocess(color, strength=0.7)
    assert jr.shape == color.shape
    assert jr.dtype == np.uint8

    tg = enh.gaussian_blur_torch(color, kernel_size=5, sigma=1.0, device="cpu")
    assert tg.shape == color.shape
    assert tg.dtype == np.uint8

    for arr in [gf, ad, sc_g, sc_c, da, out, lcn, jr, tg]:
        assert np.isfinite(np.asarray(arr, dtype=np.float32)).all()

