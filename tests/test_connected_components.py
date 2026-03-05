from __future__ import annotations

import numpy as np


def test_filter_small_components_removes_specks() -> None:
    from pyimgano.postprocess.connected_components import filter_small_components

    m = np.zeros((10, 10), dtype=np.float32)
    # Big component (area 4)
    m[1:3, 1:3] = 0.9
    # Tiny speck (area 1)
    m[8, 8] = 0.9

    filtered, comps = filter_small_components(m, threshold=0.5, min_area=2)
    assert filtered.shape == m.shape
    assert filtered.dtype == np.float32

    # Speck removed, big component kept.
    assert float(filtered[8, 8]) == 0.0
    assert float(filtered[1, 1]) > 0.0
    assert len(comps) == 1
    assert comps[0].area == 4
    assert comps[0].score is not None
