import numpy as np


def test_color_hist_extractor_shapes_and_normalization() -> None:
    from pyimgano.features.color_hist import ColorHistogramExtractor

    rng = np.random.RandomState(0)
    img = (rng.rand(32, 32, 3) * 255).astype(np.uint8)

    ext = ColorHistogramExtractor(colorspace="hsv", bins=(8, 8, 8))
    out = ext.extract([img, img])

    assert out.shape == (2, 24)
    assert np.all(np.isfinite(out))
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-5)
