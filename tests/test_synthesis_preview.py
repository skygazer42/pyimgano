from __future__ import annotations

import numpy as np


def test_make_preview_grid_smoke() -> None:
    from pyimgano.synthesis.preview import make_preview_grid

    imgs = [np.full((16, 20, 3), i * 10, dtype=np.uint8) for i in range(5)]
    masks = []
    for i in range(5):
        m = np.zeros((16, 20), dtype=np.uint8)
        if i % 2 == 0:
            m[4:12, 6:14] = 255
        masks.append(m)

    grid = make_preview_grid(imgs, masks=masks, cols=3)
    assert grid.ndim == 3
    assert grid.shape[2] == 3
    assert grid.dtype == np.uint8
    assert grid.shape[0] > 0 and grid.shape[1] > 0

