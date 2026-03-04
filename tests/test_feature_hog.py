import numpy as np
import pytest

pytest.importorskip("skimage")


def test_hog_extractor_shapes_and_finite() -> None:
    from pyimgano.features.hog import HOGExtractor

    rng = np.random.RandomState(0)
    img = (rng.rand(64, 64, 3) * 255).astype(np.uint8)

    ext = HOGExtractor(resize_hw=(64, 64))
    out = ext.extract([img, img])

    assert out.shape[0] == 2
    assert out.shape[1] > 0
    assert np.all(np.isfinite(out))
