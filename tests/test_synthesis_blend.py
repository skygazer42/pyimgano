from __future__ import annotations

import numpy as np


def test_alpha_blend_basic() -> None:
    from pyimgano.synthesis.blend import alpha_blend

    base = np.zeros((10, 10, 3), dtype=np.uint8)
    overlay = np.zeros((10, 10, 3), dtype=np.uint8)
    overlay[:, :, 2] = 255

    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:8, 2:8] = 255

    out = alpha_blend(base, overlay, mask, alpha=0.5)
    assert out.shape == base.shape
    assert out.dtype == np.uint8
    # Outside mask remains base.
    assert int(np.max(out[0:2, :, :])) == 0
    # Inside mask has red-ish values.
    assert int(np.max(out[3:7, 3:7, 2])) > 0


def test_poisson_blend_smoke() -> None:
    from pyimgano.synthesis.blend import poisson_blend

    base = np.full((32, 32, 3), 80, dtype=np.uint8)
    overlay = np.array(base, copy=True)
    overlay[12:20, 12:20, :] = 255
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:24, 8:24] = 255

    out = poisson_blend(base, overlay, mask, center_xy=(16, 16))
    assert out.shape == base.shape
    assert out.dtype == np.uint8
    # Expect some change in the masked region.
    assert int(np.mean(out[10:22, 10:22])) != int(np.mean(base[10:22, 10:22]))
