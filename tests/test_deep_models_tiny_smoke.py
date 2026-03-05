from __future__ import annotations

import numpy as np


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    return True


def test_deep_models_support_tiny_mode_smoke() -> None:
    import pytest

    if not _torch_available():
        pytest.skip("torch is not installed")

    from pyimgano.models.registry import create_model

    rng = np.random.default_rng(0)
    train = [rng.integers(0, 255, size=(32, 32, 3), dtype=np.uint8) for _ in range(4)]
    test = [rng.integers(0, 255, size=(32, 32, 3), dtype=np.uint8) for _ in range(2)]

    for name in ["ae_resnet_unet", "vae_conv"]:
        det = create_model(
            name,
            contamination=0.25,
            epochs=1,
            batch_size=2,
            lr=1e-3,
            device="cpu",
            random_state=0,
            verbose=0,
            tiny=True,
            image_size=32,
        )
        det.fit(train)
        scores = np.asarray(det.decision_function(test), dtype=np.float64).reshape(-1)
        assert scores.shape == (2,)
        assert np.all(np.isfinite(scores))
