from __future__ import annotations

import numpy as np


def test_synthesis_warp_preset_registry_and_smoke() -> None:
    from pyimgano.synthesis.presets import get_preset_names, make_preset

    assert "warp" in get_preset_names()

    h, w = 72, 96
    yy, xx = np.meshgrid(
        np.linspace(0.0, 1.0, num=h, dtype=np.float32),
        np.linspace(0.0, 1.0, num=w, dtype=np.float32),
        indexing="ij",
    )
    base = (0.4 + 0.25 * np.sin(2.0 * np.pi * (3.0 * xx + 2.0 * yy)) + 0.25 * xx).astype(
        np.float32
    )
    rgb = np.stack([base, np.clip(base * 0.95 + 0.02, 0.0, 1.0), np.roll(base, 1, axis=1)], axis=-1)
    img = (np.clip(rgb, 0.0, 1.0) * 255.0).round().astype(np.uint8)

    fn = make_preset("warp")
    rng = np.random.default_rng(123)
    out = fn(img, rng)

    assert out.overlay_u8.shape == img.shape
    assert out.overlay_u8.dtype == np.uint8
    assert out.mask_u8.shape == img.shape[:2]
    assert out.mask_u8.dtype == np.uint8
    assert int(np.sum(out.mask_u8 > 0)) > 0
    assert out.meta.get("preset") == "warp"

    # Warp should change pixel arrangement.
    assert not np.array_equal(out.overlay_u8, img)
