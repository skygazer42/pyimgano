from __future__ import annotations

import numpy as np


def test_normalize_extractor_l2_normalizes_rows() -> None:
    from pyimgano.features import create_feature_extractor

    rng = np.random.RandomState(0)
    X = rng.normal(size=(20, 8)).astype(np.float32)

    ext = create_feature_extractor("normalize", l2=True)
    Z = np.asarray(ext.extract(X), dtype=np.float64)

    assert Z.shape == X.shape
    norms = np.linalg.norm(Z, axis=1)
    assert np.all(np.isfinite(norms))
    assert np.allclose(norms, 1.0, atol=1e-6)


def test_normalize_extractor_power_changes_distribution() -> None:
    from pyimgano.features import create_feature_extractor

    X = np.array([[1.0, 4.0, 9.0]], dtype=np.float32)
    ext = create_feature_extractor("normalize", l2=False, power=0.5)
    Z = np.asarray(ext.extract(X), dtype=np.float64)

    assert Z.shape == X.shape
    assert np.all(Z >= 0.0)
    assert not np.allclose(Z, X)

