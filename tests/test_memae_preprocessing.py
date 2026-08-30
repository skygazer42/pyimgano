from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def test_memae_uint8_and_unit_float_inputs_are_equivalent() -> None:
    from pyimgano.models.memae import _preprocess_memae_images

    uint8 = np.array([[[[0], [64]], [[128], [255]]]], dtype=np.uint8)
    unit_float = uint8.astype(np.float32) / 255.0

    torch.testing.assert_close(
        _preprocess_memae_images(uint8),
        _preprocess_memae_images(unit_float),
    )


def test_memae_rejects_ambiguous_out_of_range_float_inputs() -> None:
    from pyimgano.models.memae import _preprocess_memae_images

    with pytest.raises(ValueError, match=r"\[0, 1\].*\[0, 255\]"):
        _preprocess_memae_images(np.full((1, 2, 2, 1), 256.0, dtype=np.float32))
