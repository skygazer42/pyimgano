from __future__ import annotations

import numpy as np


def test_make_preset_mixture_smoke_and_meta() -> None:
    from pyimgano.synthesis.presets import make_preset_mixture

    img = np.full((48, 64, 3), 128, dtype=np.uint8)
    fn = make_preset_mixture(["scratch", "stain", "tape"])

    rng = np.random.default_rng(0)
    out = fn(img, rng)

    assert out.overlay_u8.shape == img.shape
    assert out.overlay_u8.dtype == np.uint8
    assert out.mask_u8.shape == img.shape[:2]
    assert out.mask_u8.dtype == np.uint8
    assert isinstance(out.meta, dict)
    assert out.meta.get("preset") in {"scratch", "stain", "tape"}
    assert out.meta.get("preset_mixture") == ["scratch", "stain", "tape"]


def test_make_preset_mixture_is_deterministic_for_fixed_rng() -> None:
    from pyimgano.synthesis.presets import make_preset_mixture

    img = np.full((64, 64, 3), 90, dtype=np.uint8)
    fn = make_preset_mixture(["scratch", "texture"])

    r1 = np.random.default_rng(123)
    r2 = np.random.default_rng(123)
    o1 = fn(img, r1)
    o2 = fn(img, r2)

    assert np.array_equal(o1.overlay_u8, o2.overlay_u8)
    assert np.array_equal(o1.mask_u8, o2.mask_u8)
    assert o1.meta.get("preset") == o2.meta.get("preset")
