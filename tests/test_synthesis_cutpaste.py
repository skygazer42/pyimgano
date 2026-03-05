from __future__ import annotations

import numpy as np


def test_cutpaste_normal_and_scar_smoke() -> None:
    from pyimgano.synthesis.cutpaste import cutpaste

    img = np.full((64, 64, 3), 120, dtype=np.uint8)

    rng = np.random.default_rng(0)
    out, mask = cutpaste(img, rng=rng, variant="normal")
    assert out.shape == img.shape
    assert out.dtype == np.uint8
    assert mask.shape == (64, 64)
    assert mask.dtype == np.uint8
    assert int(np.sum(mask > 0)) > 0

    rng = np.random.default_rng(1)
    out2, mask2 = cutpaste(img, rng=rng, variant="scar")
    assert out2.shape == img.shape
    assert int(np.sum(mask2 > 0)) > 0


def test_cutpaste_determinism() -> None:
    from pyimgano.synthesis.cutpaste import cutpaste

    img = np.full((32, 48, 3), 90, dtype=np.uint8)
    rng1 = np.random.default_rng(123)
    o1, m1 = cutpaste(img, rng=rng1, variant="normal")

    rng2 = np.random.default_rng(123)
    o2, m2 = cutpaste(img, rng=rng2, variant="normal")

    assert np.array_equal(o1, o2)
    assert np.array_equal(m1, m2)
