import numpy as np
import pytest


def test_feature_registry_lists_and_creates_identity() -> None:
    from pyimgano.features import create_feature_extractor, feature_info, list_feature_extractors

    names = list_feature_extractors()
    assert "identity" in names

    ext = create_feature_extractor("identity")
    out = ext.extract([[1.0, 2.0], [3.0, 4.0]])
    assert isinstance(out, np.ndarray)
    assert out.shape == (2, 2)

    info = feature_info("identity")
    assert info["name"] == "identity"
    assert "signature" in info


def test_feature_registry_tag_filtering() -> None:
    from pyimgano.features import list_feature_extractors

    names = list_feature_extractors(tags=["embeddings"])
    assert "identity" in names


def test_feature_registry_rejects_duplicate_registration() -> None:
    from pyimgano.features.registry import FeatureRegistry

    class _Dummy:
        def __init__(self, *, k: int = 1) -> None:
            self.k = int(k)

    r = FeatureRegistry()
    r.register("x", _Dummy)
    with pytest.raises(KeyError, match="already exists"):
        r.register("x", _Dummy)

