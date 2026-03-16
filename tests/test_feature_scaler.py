import numpy as np


def test_standard_scaler_extractor_centers_features() -> None:
    from pyimgano.features.scaler import StandardScalerExtractor

    rng = np.random.default_rng(0)
    X = rng.normal(loc=10.0, scale=3.0, size=(50, 4))

    ext = StandardScalerExtractor()
    ext.fit(X)
    Z = ext.extract(X)

    assert Z.shape == X.shape
    # Approximately zero mean per feature
    assert np.allclose(Z.mean(axis=0), 0.0, atol=1e-5)
