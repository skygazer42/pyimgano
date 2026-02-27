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


def test_base_feature_extractor_get_set_params() -> None:
    from pyimgano.features.base import BaseFeatureExtractor

    class Dummy(BaseFeatureExtractor):
        def __init__(self, *, k: int = 1) -> None:
            self.k = int(k)

        def extract(self, inputs):
            inputs = list(inputs)
            return np.zeros((len(inputs), self.k), dtype=np.float32)

    ext = Dummy(k=3)
    assert ext.get_params()["k"] == 3
    ext.set_params(k=5)
    assert ext.k == 5
