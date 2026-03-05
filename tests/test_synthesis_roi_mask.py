from __future__ import annotations

import numpy as np


def test_synthesis_roi_mask_can_force_empty_and_skip() -> None:
    from pyimgano.synthesis.synthesizer import AnomalySynthesizer, SynthSpec

    img = np.full((32, 32, 3), 100, dtype=np.uint8)
    roi = np.zeros((32, 32), dtype=np.uint8)  # no allowed region

    syn = AnomalySynthesizer(
        SynthSpec(preset="scratch", probability=1.0, blend="alpha", alpha=1.0, max_tries=3)
    )
    res = syn(img, seed=0, roi_mask=roi)
    assert res.label == 0
    assert int(np.sum(res.mask_u8 > 0)) == 0
