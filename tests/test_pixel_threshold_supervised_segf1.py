from __future__ import annotations

import numpy as np
import pytest


def test_supervised_segf1_threshold_picks_perfect_cutoff() -> None:
    from pyimgano.calibration.pixel_threshold_supervised import (
        calibrate_supervised_segf1_threshold,
    )

    maps = np.asarray([[[0.1, 0.2], [0.8, 0.9]]], dtype=np.float32)
    masks = np.asarray([[[0, 0], [1, 1]]], dtype=np.uint8)

    thr = float(calibrate_supervised_segf1_threshold(maps, masks))
    assert thr == pytest.approx(0.8, abs=1e-6)

