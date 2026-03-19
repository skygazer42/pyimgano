from __future__ import annotations

import numpy as np
import pytest


def _mean_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float32)
    bb = np.asarray(b, dtype=np.float32)
    return float(np.mean(np.abs(aa - bb)))


def test_synthesis_multiple_defects_union_mask_and_meta() -> None:
    from pyimgano.synthesis.synthesizer import AnomalySynthesizer, SynthSpec

    img = np.full((64, 64, 3), 120, dtype=np.uint8)

    single = AnomalySynthesizer(
        SynthSpec(preset="scratch", probability=1.0, blend="alpha", alpha=1.0, max_tries=5)
    )
    multi = AnomalySynthesizer(
        SynthSpec(
            preset="scratch",
            probability=1.0,
            blend="alpha",
            alpha=1.0,
            max_tries=5,
            num_defects=2,
            severity=1.0,
        )
    )

    r1 = single(img, seed=123)
    r2 = multi(img, seed=123)

    a1 = int(np.sum(r1.mask_u8 > 0))
    a2 = int(np.sum(r2.mask_u8 > 0))
    assert a1 > 0
    assert a2 >= a1

    assert np.isclose(r2.meta.get("severity"), 1.0)
    assert int(r2.meta.get("num_defects", 0)) == 2
    assert int(r2.meta.get("defects_applied", 0)) == 2


def test_synthesis_severity_scales_effect_strength() -> None:
    from pyimgano.synthesis.synthesizer import AnomalySynthesizer, SynthSpec

    img = np.full((64, 64, 3), 128, dtype=np.uint8)

    low = AnomalySynthesizer(
        SynthSpec(preset="scratch", probability=1.0, blend="alpha", alpha=1.0, severity=0.25)
    )
    high = AnomalySynthesizer(
        SynthSpec(preset="scratch", probability=1.0, blend="alpha", alpha=1.0, severity=1.0)
    )

    r_low = low(img, seed=7)
    r_high = high(img, seed=7)

    d_low = _mean_abs_diff(r_low.image_u8, img)
    d_high = _mean_abs_diff(r_high.image_u8, img)
    assert d_low < d_high
