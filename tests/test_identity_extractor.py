import numpy as np


def test_identity_extractor_extracts_numpy_matrix() -> None:
    from pyimgano.features.identity import IdentityExtractor

    ext = IdentityExtractor()
    out = ext.extract([[1.0, 2.0], [3.0, 4.0]])
    assert isinstance(out, np.ndarray)
    assert out.shape == (2, 2)
    assert np.allclose(out, np.asarray([[1.0, 2.0], [3.0, 4.0]]))


def test_legacy_identity_feature_extractor_alias() -> None:
    from pyimgano.detectors import IdentityFeatureExtractor
    from pyimgano.features.identity import IdentityExtractor

    assert IdentityFeatureExtractor is IdentityExtractor
