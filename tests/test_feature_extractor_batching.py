import numpy as np
import pytest


def test_base_feature_extractor_extract_batched_concats_rows() -> None:
    from pyimgano.features.base import BaseFeatureExtractor

    class _Counting(BaseFeatureExtractor):
        def __init__(self) -> None:
            self.calls = 0

        def extract(self, inputs):  # noqa: ANN001, ANN201 - test helper
            self.calls += 1
            xs = list(inputs)
            return np.asarray([[float(x)] for x in xs], dtype=np.float32)

    ext = _Counting()
    out = ext.extract_batched(range(10), batch_size=3)

    assert out.shape == (10, 1)
    assert ext.calls == 4  # 3+3+3+1
    assert np.allclose(out[:, 0], np.arange(10, dtype=np.float32))


def test_base_feature_extractor_extract_batched_rejects_non_positive_batch_size() -> None:
    from pyimgano.features.base import BaseFeatureExtractor

    class _Stub(BaseFeatureExtractor):
        def extract(self, inputs):  # noqa: ANN001, ANN201 - test helper
            xs = list(inputs)
            return np.asarray(xs, dtype=np.float32).reshape(-1, 1)

    ext = _Stub()
    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        ext.extract_batched([1, 2, 3], batch_size=0)

