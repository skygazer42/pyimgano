from __future__ import annotations

import numpy as np


def test_synthesis_is_deterministic_for_fixed_seed() -> None:
    from pyimgano.synthesis import AnomalySynthesizer, SynthSpec

    img = np.full((64, 64, 3), 100, dtype=np.uint8)
    syn = AnomalySynthesizer(SynthSpec(preset="stain", probability=1.0, blend="alpha", alpha=0.9))

    r1 = syn(img, seed=999)
    r2 = syn(img, seed=999)
    assert np.array_equal(r1.image_u8, r2.image_u8)
    assert np.array_equal(r1.mask_u8, r2.mask_u8)


def test_new_preset_texture_is_deterministic_for_fixed_seed() -> None:
    from pyimgano.synthesis import AnomalySynthesizer, SynthSpec

    img = np.full((64, 64, 3), 140, dtype=np.uint8)
    syn = AnomalySynthesizer(
        SynthSpec(preset="texture", probability=1.0, blend="alpha", alpha=0.85)
    )

    r1 = syn(img, seed=1234)
    r2 = syn(img, seed=1234)
    assert np.array_equal(r1.image_u8, r2.image_u8)
    assert np.array_equal(r1.mask_u8, r2.mask_u8)
