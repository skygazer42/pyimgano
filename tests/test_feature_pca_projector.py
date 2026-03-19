import numpy as np


def test_pca_projector_fit_transform_shapes() -> None:
    from pyimgano.features.pca_projector import PCAProjector

    rng = np.random.default_rng(0)
    x = rng.normal(size=(50, 10))

    ext = PCAProjector(n_components=5, random_state=0)
    ext.fit(x)
    z = ext.extract(x[:7])

    assert z.shape == (7, 5)
    assert np.all(np.isfinite(z))
