import numpy as np
import pytest

pytest.importorskip("skimage")


def test_gabor_bank_extractor_shapes_and_finite() -> None:
    from pyimgano.features.gabor import GaborBankExtractor

    rng = np.random.RandomState(0)
    img = (rng.rand(64, 64, 3) * 255).astype(np.uint8)

    ext = GaborBankExtractor(resize_hw=(64, 64), frequencies=(0.2,), thetas=(0.0, np.pi / 2))
    out = ext.extract([img, img])

    assert out.shape == (2, 4)  # 1 freq * 2 thetas * (mean,std)
    assert np.all(np.isfinite(out))
