from __future__ import annotations

import numpy as np


def test_synthesis_illumination_preset_registry_and_smoke() -> None:
    from pyimgano.synthesis.presets import get_preset_names, make_preset

    assert "illumination" in get_preset_names()

    # Use a non-uniform image so geometric/illumination changes are measurable.
    h, w = 64, 80
    yy, xx = np.meshgrid(
        np.linspace(0.0, 1.0, num=h, dtype=np.float32),
        np.linspace(0.0, 1.0, num=w, dtype=np.float32),
        indexing="ij",
    )
    base = (0.45 + 0.35 * xx + 0.15 * np.sin(2.0 * np.pi * (2.0 * yy))).astype(np.float32)
    rgb = np.stack([base, np.clip(base * 0.9 + 0.03, 0.0, 1.0), base], axis=-1)
    img = (np.clip(rgb, 0.0, 1.0) * 255.0).round().astype(np.uint8)

    fn = make_preset("illumination")
    rng = np.random.default_rng(0)
    out = fn(img, rng)

    assert out.overlay_u8.shape == img.shape
    assert out.overlay_u8.dtype == np.uint8
    assert out.mask_u8.shape == img.shape[:2]
    assert out.mask_u8.dtype == np.uint8
    assert int(np.sum(out.mask_u8 > 0)) > 0
    assert out.meta.get("preset") == "illumination"

    # Should meaningfully alter pixels.
    assert not np.array_equal(out.overlay_u8, img)
