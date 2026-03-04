import importlib.util

import numpy as np
import pytest


def test_feature_extractors_smoke_on_synthetic_images() -> None:
    from pyimgano.features import create_feature_extractor

    rng = np.random.RandomState(0)
    imgs = [(rng.rand(32, 32, 3) * 255).astype(np.uint8) for _ in range(3)]

    has_skimage = importlib.util.find_spec("skimage") is not None

    specs = [
        ("color_hist", {"colorspace": "hsv", "bins": [4, 4, 4]}),
        ("edge_stats", {}),
        ("fft_lowfreq", {"size_hw": [32, 32], "radii": [4, 8]}),
        ("patch_stats", {"grid": [2, 2], "stats": ["mean", "std"], "resize_hw": [32, 32]}),
    ]

    skimage_specs = [
        ("hog", {"resize_hw": [32, 32]}),
        ("lbp", {"n_points": 8, "radius": 1.0, "method": "uniform"}),
        ("gabor_bank", {"resize_hw": [32, 32], "frequencies": [0.2], "thetas": [0.0]}),
    ]

    if has_skimage:
        specs = skimage_specs + specs
    else:
        # Ensure these feature extractors fail with an actionable hint when scikit-image isn't installed.
        for name, kwargs in skimage_specs:
            ext = create_feature_extractor(name, **kwargs)
            with pytest.raises(ImportError) as excinfo:
                ext.extract(imgs)
            assert "pyimgano[skimage]" in str(excinfo.value)

    for name, kwargs in specs:
        ext = create_feature_extractor(name, **kwargs)
        out = np.asarray(ext.extract(imgs))
        assert out.ndim == 2
        assert out.shape[0] == len(imgs)
        assert out.shape[1] > 0
        assert np.all(np.isfinite(out))


def test_feature_extractors_smoke_on_vectors() -> None:
    from pyimgano.features import create_feature_extractor

    rng = np.random.RandomState(0)
    X = rng.normal(size=(50, 8))

    identity = create_feature_extractor("identity")
    out = identity.extract(X[:3])
    assert out.shape == (3, 8)

    scaler = create_feature_extractor("standard_scaler")
    scaler.fit(X)
    Z = scaler.extract(X[:5])
    assert Z.shape == (5, 8)
    assert np.all(np.isfinite(Z))

    pca = create_feature_extractor("pca_projector", n_components=4, random_state=0)
    pca.fit(X)
    P = pca.extract(X[:7])
    assert P.shape == (7, 4)
    assert np.all(np.isfinite(P))
