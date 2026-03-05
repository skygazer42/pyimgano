from __future__ import annotations

import numpy as np


def test_feature_extractor_protocol_runtime_check() -> None:
    from pyimgano.features.protocols import FeatureExtractor

    class Dummy:
        def extract(self, inputs):
            inputs = list(inputs)
            return np.zeros((len(inputs), 3), dtype=np.float32)

    assert isinstance(Dummy(), FeatureExtractor)


def test_fittable_feature_extractor_protocol_runtime_check() -> None:
    from pyimgano.features.protocols import FittableFeatureExtractor

    class Dummy:
        def fit(self, inputs, y=None):
            return self

        def extract(self, inputs):
            inputs = list(inputs)
            return np.ones((len(inputs), 2), dtype=np.float32)

    assert isinstance(Dummy(), FittableFeatureExtractor)
