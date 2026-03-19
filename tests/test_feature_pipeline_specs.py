from __future__ import annotations

import numpy as np


def test_multi_extractor_accepts_registry_specs() -> None:
    from pyimgano.features import create_feature_extractor

    ext = create_feature_extractor("multi", extractors=["identity", "identity"])

    x = np.arange(12, dtype=np.float32).reshape(3, 4)
    z = np.asarray(ext.extract(x), dtype=np.float32)

    assert z.shape == (3, 8)
    assert np.allclose(z[:, :4], x)
    assert np.allclose(z[:, 4:], x)
