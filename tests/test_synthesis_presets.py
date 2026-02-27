from __future__ import annotations

import numpy as np


def test_presets_registry_and_smoke_apply() -> None:
    from pyimgano.synthesis.presets import get_preset_names, make_preset

    names = get_preset_names()
    assert "scratch" in names
    assert "stain" in names
    # Industrial additions
    assert "rust" in names
    assert "oil" in names
    assert "crack" in names

    img = np.full((48, 48, 3), 120, dtype=np.uint8)
    rng = np.random.default_rng(0)

    for name in names:
        fn = make_preset(name)
        out = fn(img, rng)
        assert out.overlay_u8.shape == img.shape
        assert out.overlay_u8.dtype == np.uint8
        assert out.mask_u8.shape == img.shape[:2]
        assert out.mask_u8.dtype == np.uint8
        assert isinstance(out.meta, dict)
