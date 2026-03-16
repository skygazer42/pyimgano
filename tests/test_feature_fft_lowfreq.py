import numpy as np


def test_fft_lowfreq_extractor_shapes_and_bounds() -> None:
    from pyimgano.features.fft_lowfreq import FFTLowFreqExtractor

    rng = np.random.default_rng(0)
    img = (rng.random((64, 64, 3)) * 255).astype(np.uint8)

    ext = FFTLowFreqExtractor(size_hw=(64, 64), radii=(4, 8))
    out = ext.extract([img, img])

    assert out.shape == (2, 2)
    assert np.all(np.isfinite(out))
    assert np.all((out >= 0.0) & (out <= 1.0))
