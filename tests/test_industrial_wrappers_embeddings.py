from __future__ import annotations

import numpy as np


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    return True


def test_embedding_industrial_wrappers_are_registered() -> None:
    from pyimgano.models import list_models

    names = set(list_models())
    assert "vision_resnet18_ecod" in names
    assert "vision_resnet18_iforest" in names
    assert "vision_resnet18_knn" in names
    assert "vision_resnet18_torch_ae" in names


def test_embedding_industrial_wrappers_accept_identity_extractor_on_vectors() -> None:
    import pytest

    from pyimgano.models import create_model

    rng = np.random.default_rng(0)
    X = [rng.normal(size=(16,)).astype(np.float32) for _ in range(80)]

    for name in ["vision_resnet18_ecod", "vision_resnet18_iforest", "vision_resnet18_knn"]:
        det = create_model(name, contamination=0.2, embedding_extractor="identity")
        det.fit(X)
        scores = np.asarray(det.decision_function(X[:5]), dtype=np.float64).reshape(-1)
        assert scores.shape == (5,)
        assert np.all(np.isfinite(scores))

    if not _torch_available():
        pytest.skip("torch is not installed (required for core_torch_autoencoder)")

    det = create_model(
        "vision_resnet18_torch_ae",
        contamination=0.25,
        embedding_extractor="identity",
        core_kwargs={
            "epochs": 2,
            "batch_size": 16,
            "lr": 1e-3,
            "device": "cpu",
            "random_state": 0,
            "hidden_dims": (12, 4),
        },
    )
    det.fit(X)
    scores = np.asarray(det.decision_function(X[:5]), dtype=np.float64).reshape(-1)
    assert scores.shape == (5,)
    assert np.all(np.isfinite(scores))

