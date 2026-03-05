from __future__ import annotations

from pathlib import Path

import numpy as np


def test_anomaly_synthesizer_determinism_and_nonempty_mask() -> None:
    from pyimgano.synthesis.synthesizer import AnomalySynthesizer, SynthSpec

    img = np.full((64, 64, 3), 120, dtype=np.uint8)
    syn = AnomalySynthesizer(SynthSpec(preset="scratch", probability=1.0, blend="alpha", alpha=1.0))

    r1 = syn(img, seed=123)
    r2 = syn(img, seed=123)
    assert r1.label == 1
    assert r1.mask_u8.shape == (64, 64)
    assert int(np.sum(r1.mask_u8 > 0)) > 0
    assert np.array_equal(r1.image_u8, r2.image_u8)
    assert np.array_equal(r1.mask_u8, r2.mask_u8)


def test_anomaly_synthesizer_respects_roi_mask() -> None:
    from pyimgano.synthesis.synthesizer import AnomalySynthesizer, SynthSpec

    img = np.full((32, 32, 3), 80, dtype=np.uint8)
    roi = np.zeros((32, 32), dtype=np.uint8)
    roi[:, :16] = 255  # only left half allowed

    syn = AnomalySynthesizer(SynthSpec(preset="pit", probability=1.0, blend="alpha", alpha=1.0))
    res = syn(img, seed=7, roi_mask=roi)
    assert res.mask_u8.shape == (32, 32)

    # No mask pixels should be outside ROI.
    assert int(np.sum((res.mask_u8 > 0) & (roi == 0))) == 0


def test_anomaly_synthesizer_can_skip_by_probability(tmp_path: Path) -> None:
    from pyimgano.synthesis.synthesizer import AnomalySynthesizer, SynthSpec

    _ = tmp_path
    img = np.full((16, 16, 3), 10, dtype=np.uint8)
    syn = AnomalySynthesizer(SynthSpec(preset="scratch", probability=0.0))
    res = syn(img, seed=0)
    assert res.label == 0
    assert int(np.sum(res.mask_u8 > 0)) == 0
