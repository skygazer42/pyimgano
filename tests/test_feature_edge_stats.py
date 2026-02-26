import numpy as np


def test_edge_stats_extractor_shapes_and_finite() -> None:
    from pyimgano.features.edge_stats import EdgeStatsExtractor

    rng = np.random.RandomState(0)
    img = (rng.rand(64, 64, 3) * 255).astype(np.uint8)

    ext = EdgeStatsExtractor()
    out = ext.extract([img, img])

    assert out.shape == (2, 6)
    assert np.all(np.isfinite(out))

