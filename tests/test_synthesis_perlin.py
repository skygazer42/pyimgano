from __future__ import annotations

import numpy as np


def test_perlin_noise_shape_range_and_determinism() -> None:
    from pyimgano.synthesis.perlin import perlin_noise_2d

    rng1 = np.random.default_rng(123)
    n1 = perlin_noise_2d((32, 48), (4, 6), rng=rng1)
    assert n1.shape == (32, 48)
    assert n1.dtype == np.float32
    assert float(np.min(n1)) >= 0.0
    assert float(np.max(n1)) <= 1.0

    rng2 = np.random.default_rng(123)
    n2 = perlin_noise_2d((32, 48), (4, 6), rng=rng2)
    assert np.allclose(n1, n2)


def test_fractal_perlin_noise_smoke() -> None:
    from pyimgano.synthesis.perlin import fractal_perlin_noise_2d

    rng = np.random.default_rng(7)
    n = fractal_perlin_noise_2d((40, 40), (3, 3), rng=rng, octaves=4)
    assert n.shape == (40, 40)
    assert n.dtype == np.float32
    assert float(np.min(n)) >= 0.0
    assert float(np.max(n)) <= 1.0

