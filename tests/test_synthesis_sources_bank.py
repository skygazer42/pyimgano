from __future__ import annotations

import numpy as np


def test_texture_source_bank_samples_overlay_with_target_shape() -> None:
    from pyimgano.synthesis.sources import TextureSourceBank

    rng = np.random.default_rng(0)
    sources = [
        rng.integers(0, 255, size=(40, 50, 3), dtype=np.uint8),
        rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8),
    ]

    bank = TextureSourceBank(sources)
    overlay = bank.sample_overlay((32, 48), rng=rng)

    assert overlay.shape == (32, 48, 3)
    assert overlay.dtype == np.uint8

