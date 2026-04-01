from __future__ import annotations

import numpy as np
import pytest


def test_preprocess_imagenet_batch_converts_hwc_uint8_to_normalized_chw_tensor() -> None:
    torch = pytest.importorskip("torch")

    from pyimgano.models._imagenet_preprocess import preprocess_imagenet_batch

    batch = np.zeros((1, 2, 2, 3), dtype=np.uint8)
    out = preprocess_imagenet_batch(batch)

    assert isinstance(out, torch.Tensor)
    assert tuple(out.shape) == (1, 3, 2, 2)
    assert out.dtype == torch.float32
    assert float(out[0, 0, 0, 0]) == pytest.approx((0.0 - 0.485) / 0.229)
    assert float(out[0, 1, 0, 0]) == pytest.approx((0.0 - 0.456) / 0.224)
    assert float(out[0, 2, 0, 0]) == pytest.approx((0.0 - 0.406) / 0.225)
