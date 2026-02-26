import numpy as np


def test_patch_stats_extractor_shapes_and_finite() -> None:
    from pyimgano.features.patch_stats import PatchStatsExtractor

    rng = np.random.RandomState(0)
    img = (rng.rand(64, 64, 3) * 255).astype(np.uint8)

    ext = PatchStatsExtractor(grid=(2, 3), stats=("mean", "std"))
    out = ext.extract([img, img])

    assert out.shape == (2, 2 * 3 * 2)
    assert np.all(np.isfinite(out))

