from __future__ import annotations

import numpy as np

from pyimgano.defects.pixel_threshold import resolve_pixel_threshold


def test_resolve_pixel_threshold_explicit_threshold_has_provenance() -> None:
    thr, prov = resolve_pixel_threshold(
        pixel_threshold=0.7,
        pixel_threshold_strategy="normal_pixel_quantile",
        infer_config_pixel_threshold=None,
        calibration_maps=None,
    )
    assert thr == 0.7
    assert prov["source"] == "explicit"
    assert prov["method"] == "fixed"


def test_resolve_pixel_threshold_quantile_has_q_and_calibration_count() -> None:
    m1 = np.zeros((2, 2), dtype=np.float32)
    m2 = np.ones((2, 2), dtype=np.float32)
    thr, prov = resolve_pixel_threshold(
        pixel_threshold=None,
        pixel_threshold_strategy="normal_pixel_quantile",
        infer_config_pixel_threshold=None,
        calibration_maps=[m1, m2],
        pixel_normal_quantile=0.5,
    )
    assert thr == 0.5
    assert prov["method"] == "normal_pixel_quantile"
    assert prov["q"] == 0.5
    assert prov["calibration_map_count"] == 2

