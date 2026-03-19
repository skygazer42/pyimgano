from __future__ import annotations

import numpy as np


def test_normalize_extractor_l2_normalizes_rows() -> None:
    from pyimgano.features import create_feature_extractor

    rng = np.random.default_rng(0)
    x = rng.normal(size=(20, 8)).astype(np.float32)

    ext = create_feature_extractor("normalize", l2=True)
    z = np.asarray(ext.extract(x), dtype=np.float64)

    assert z.shape == x.shape
    norms = np.linalg.norm(z, axis=1)
    assert np.all(np.isfinite(norms))
    assert np.allclose(norms, 1.0, atol=1e-6)


def test_normalize_extractor_power_changes_distribution() -> None:
    from pyimgano.features import create_feature_extractor

    x = np.array([[1.0, 4.0, 9.0]], dtype=np.float32)
    ext = create_feature_extractor("normalize", l2=False, power=0.5)
    z = np.asarray(ext.extract(x), dtype=np.float64)

    assert z.shape == x.shape
    assert np.all(z >= 0.0)
    assert not np.allclose(z, x)
