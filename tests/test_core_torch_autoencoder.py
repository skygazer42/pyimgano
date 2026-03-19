from __future__ import annotations

import numpy as np


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    return True


def test_core_torch_autoencoder_fit_predict_smoke() -> None:
    import pytest

    if not _torch_available():
        pytest.skip("torch is not installed")

    from pyimgano.models.registry import create_model

    rng = np.random.default_rng(0)
    x = rng.normal(size=(60, 12)).astype(np.float64)

    det = create_model(
        "core_torch_autoencoder",
        contamination=0.2,
        hidden_dims=(16, 6),
        activation="relu",
        dropout=0.0,
        epochs=3,
        batch_size=16,
        lr=1e-3,
        device="cpu",
        preprocessing=True,
        random_state=0,
    )
    det.fit(x)
    scores = np.asarray(det.decision_function(x[:8]), dtype=np.float64).reshape(-1)
    preds = np.asarray(det.predict(x[:8]), dtype=np.int64).reshape(-1)
    assert scores.shape == (8,)
    assert preds.shape == (8,)
    assert np.all(np.isfinite(scores))
    assert set(np.unique(preds)).issubset({0, 1})


def test_vision_torch_autoencoder_supports_identity_extractor_on_vectors() -> None:
    import pytest

    if not _torch_available():
        pytest.skip("torch is not installed")

    from pyimgano.models.registry import create_model

    rng = np.random.default_rng(1)
    x = [rng.normal(size=(10,)).astype(np.float32) for _ in range(80)]

    det = create_model(
        "vision_torch_autoencoder",
        contamination=0.25,
        feature_extractor="identity",
        hidden_dims=(12, 4),
        epochs=2,
        batch_size=16,
        lr=1e-3,
        device="cpu",
        preprocessing=True,
        random_state=0,
    )
    det.fit(x)
    scores = np.asarray(det.decision_function(x[:5]), dtype=np.float64).reshape(-1)
    assert scores.shape == (5,)
    assert np.all(np.isfinite(scores))
