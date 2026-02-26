from __future__ import annotations

import numpy as np
import pytest


def test_frequency_filter_rejects_unknown_filter_type() -> None:
    from pyimgano.preprocessing.advanced_operations import frequency_filter

    img = np.zeros((32, 32), dtype=np.uint8)
    with pytest.raises(ValueError, match="filter_type"):
        _ = frequency_filter(img, filter_type="not-a-real-filter", cutoff_frequency=5.0)


@pytest.mark.parametrize("filter_type", ["lowpass", "highpass", "bandpass", "bandstop"])
def test_frequency_filter_preserves_shape_and_dtype(filter_type: str) -> None:
    from pyimgano.preprocessing.advanced_operations import frequency_filter

    img = np.zeros((32, 32), dtype=np.uint8)
    img[8:24, 8:24] = 255

    out = frequency_filter(img, filter_type=filter_type, cutoff_frequency=5.0)
    assert out.shape == img.shape
    assert out.dtype == np.uint8
    assert np.isfinite(out.astype(np.float32)).all()

