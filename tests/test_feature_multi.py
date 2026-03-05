import numpy as np


def test_multi_extractor_concats_feature_matrices() -> None:
    from pyimgano.features.identity import IdentityExtractor
    from pyimgano.features.multi import MultiExtractor

    class _Const:
        def extract(self, inputs):  # noqa: ANN001, ANN201 - test helper
            n = len(list(inputs))
            return np.ones((n, 2), dtype=np.float32)

    X = np.arange(12, dtype=np.float32).reshape(4, 3)
    ext = MultiExtractor([IdentityExtractor(), _Const()])
    out = ext.extract(X)

    assert out.shape == (4, 5)
    assert np.allclose(out[:, :3], X)
    assert np.allclose(out[:, 3:], 1.0)
