from __future__ import annotations

import numpy as np


def test_base_feature_extractor_fit_extract() -> None:
    from pyimgano.features.base import BaseFeatureExtractor

    class Dummy(BaseFeatureExtractor):
        def extract(self, inputs):
            inputs = list(inputs)
            return np.zeros((len(inputs), 4), dtype=np.float32)

    ext = Dummy()
    X = ["a", "b", "c"]
    out = ext.fit_extract(X)
    assert out.shape == (3, 4)

