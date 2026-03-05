from __future__ import annotations

import numpy as np


def test_anisotropic_diffusion_runs_and_preserves_contract() -> None:
    from pyimgano.preprocessing.anisotropic_diffusion import anisotropic_diffusion

    img = np.zeros((32, 32), dtype=np.uint8)
    img[8:24, 8:24] = 255

    out = anisotropic_diffusion(img, niter=5, kappa=30.0, gamma=0.1, option=1)
    assert out.shape == img.shape
    assert out.dtype == np.uint8
    assert int(out.min()) >= 0
    assert int(out.max()) <= 255
