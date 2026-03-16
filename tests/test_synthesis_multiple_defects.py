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

    assert np.isclose(float(r2.meta.get("severity", 0.0)), 1.0)
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


def test_synthesizer_no_seed_path_uses_explicit_seeded_rng_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    import pyimgano.utils.random_state as random_state_module
    from pyimgano.synthesis.synthesizer import AnomalySynthesizer, SynthSpec

    img = np.full((32, 32, 3), 120, dtype=np.uint8)
    synth = AnomalySynthesizer(
        SynthSpec(preset="scratch", probability=0.0, blend="alpha", alpha=1.0)
    )

    observed_seeds: list[int | None] = []
    original_default_rng = random_state_module.np.random.default_rng

    monkeypatch.setattr(random_state_module.os, "urandom", lambda n: b"\x07" * n)

    def _tracking_default_rng(seed=None):
        observed_seeds.append(seed)
        return original_default_rng(seed)

    monkeypatch.setattr(random_state_module.np.random, "default_rng", _tracking_default_rng)

    synth.synthesize(img)

    assert observed_seeds
    assert observed_seeds[0] is not None
