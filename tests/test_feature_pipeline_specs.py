from __future__ import annotations

import numpy as np


def test_multi_extractor_accepts_registry_specs() -> None:
    from pyimgano.features import create_feature_extractor

    ext = create_feature_extractor("multi", extractors=["identity", "identity"])

    X = np.arange(12, dtype=np.float32).reshape(3, 4)
    Z = np.asarray(ext.extract(X), dtype=np.float32)

    assert Z.shape == (3, 8)
    assert np.allclose(Z[:, :4], X)
    assert np.allclose(Z[:, 4:], X)

