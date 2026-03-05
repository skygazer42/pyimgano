from __future__ import annotations

import numpy as np


def test_morphology_open_close_float01_smoke() -> None:
    from pyimgano.postprocess.morphology import close_float01, open_float01

    m = np.zeros((32, 32), dtype=np.float32)
    m[10:12, 10:12] = 1.0
    m[20, 20] = 1.0  # tiny speck

    opened = open_float01(m, ksize=3)
    closed = close_float01(m, ksize=3)

    assert opened.shape == m.shape
    assert closed.shape == m.shape
    assert opened.dtype == np.float32
    assert closed.dtype == np.float32
    assert float(opened.min()) >= 0.0 and float(opened.max()) <= 1.0
    assert float(closed.min()) >= 0.0 and float(closed.max()) <= 1.0


def test_morphology_u8_mask_smoke() -> None:
    from pyimgano.postprocess.morphology import morph_u8

    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[3:5, 3:5] = 255
    out = morph_u8(mask, op="open", ksize=3)
    assert out.shape == mask.shape
    assert out.dtype == np.uint8
