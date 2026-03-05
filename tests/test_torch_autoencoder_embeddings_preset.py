from __future__ import annotations

import numpy as np


def test_vision_embedding_torch_autoencoder_defaults_are_safe() -> None:
    import pyimgano.models  # noqa: F401 - registry population side effects
    from pyimgano.models.registry import create_model

    det = create_model("vision_embedding_torch_autoencoder", device="cpu")

    # Ensure the default embedding extractor is safe-by-default (no implicit downloads).
    from pyimgano.features.torchvision_backbone import TorchvisionBackboneExtractor

    assert isinstance(det.feature_extractor, TorchvisionBackboneExtractor)
    assert det.feature_extractor.pretrained is False


def test_vision_embedding_torch_autoencoder_smoke_on_vectors() -> None:
    import pyimgano.models  # noqa: F401
    from pyimgano.models.registry import create_model

    rng = np.random.default_rng(0)
    X_train = rng.standard_normal(size=(32, 16)).astype(np.float32)
    X_test = rng.standard_normal(size=(5, 16)).astype(np.float32)

    det = create_model(
        "vision_embedding_torch_autoencoder",
        device="cpu",
        embedding_extractor="identity",
        ae_kwargs={
            "hidden_dims": (8,),
            "epochs": 1,
            "batch_size": 8,
            "lr": 1e-3,
            "device": "cpu",
            "preprocessing": True,
            "random_state": 0,
        },
        contamination=0.25,
    )
    det.fit(X_train)
    scores = np.asarray(det.decision_function(X_test), dtype=np.float64)
    assert scores.shape == (X_test.shape[0],)
    assert np.isfinite(scores).all()
