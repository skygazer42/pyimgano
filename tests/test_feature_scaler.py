import numpy as np


def test_standard_scaler_extractor_centers_features() -> None:
    from pyimgano.features.scaler import StandardScalerExtractor

    rng = np.random.default_rng(0)
    x = rng.normal(loc=10.0, scale=3.0, size=(50, 4))

    ext = StandardScalerExtractor()
    ext.fit(x)
    z = ext.extract(x)

    assert z.shape == x.shape
    # Approximately zero mean per feature
    assert np.allclose(z.mean(axis=0), 0.0, atol=1e-5)
