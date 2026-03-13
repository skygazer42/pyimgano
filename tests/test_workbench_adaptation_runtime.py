from __future__ import annotations

import numpy as np

from pyimgano.workbench.adaptation_runtime import apply_tiling, build_postprocess
from pyimgano.workbench.adaptation_types import MapPostprocessConfig, TilingConfig


def test_adaptation_runtime_apply_tiling_passthrough_when_disabled() -> None:
    detector = object()

    wrapped = apply_tiling(detector, TilingConfig(tile_size=None))

    assert wrapped is detector


def test_adaptation_runtime_build_postprocess_builds_callable() -> None:
    cfg = MapPostprocessConfig(
        normalize=True,
        normalize_method="percentile",
        percentile_range=(10.0, 90.0),
        gaussian_sigma=0.0,
        morph_open_ksize=0,
        morph_close_ksize=0,
        component_threshold=None,
        min_component_area=0,
    )

    post = build_postprocess(cfg)

    assert post is not None
    out = post(np.arange(16, dtype=np.float32).reshape(4, 4))
    assert out.shape == (4, 4)
