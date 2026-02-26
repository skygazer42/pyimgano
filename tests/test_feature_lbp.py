import numpy as np


def test_lbp_extractor_shapes_and_normalized_hist() -> None:
    from pyimgano.features.lbp import LBPExtractor

    rng = np.random.RandomState(0)
    img = (rng.rand(64, 64, 3) * 255).astype(np.uint8)

    ext = LBPExtractor(n_points=8, radius=1.0, method="uniform")
    out = ext.extract([img, img])

    assert out.shape[0] == 2
    assert out.shape[1] == 10  # n_points + 2 for uniform
    assert np.all(np.isfinite(out))
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-5)

