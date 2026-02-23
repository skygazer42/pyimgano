import numpy as np

from pyimgano.workbench.adaptation import MapPostprocessConfig, build_postprocess


def test_build_postprocess_none_returns_none():
    assert build_postprocess(None) is None


def test_build_postprocess_builds_callable():
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

    m = np.arange(16, dtype=np.float32).reshape(4, 4)
    out = post(m)
    assert out.shape == (4, 4)

