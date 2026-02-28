from __future__ import annotations

import numpy as np


def test_industrial_embedding_core_wrappers_registered() -> None:
    from pyimgano.models import list_models

    names = set(list_models())
    assert "vision_resnet18_knn_cosine" in names
    assert "vision_resnet18_mahalanobis_shrinkage" in names


def test_industrial_embedding_core_wrappers_accept_identity_extractor_on_vectors() -> None:
    import pyimgano.models  # noqa: F401
    from pyimgano.models import create_model

    rng = np.random.default_rng(0)
    X = [rng.normal(size=(16,)).astype(np.float32) for _ in range(50)]

    for name in ["vision_resnet18_knn_cosine", "vision_resnet18_mahalanobis_shrinkage"]:
        det = create_model(name, contamination=0.2, embedding_extractor="identity")
        det.fit(X)
        scores = np.asarray(det.decision_function(X[:7]), dtype=np.float64).reshape(-1)
        assert scores.shape == (7,)
        assert np.all(np.isfinite(scores))

