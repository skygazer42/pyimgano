from __future__ import annotations

import numpy as np


def test_core_knn_cosine_smoke() -> None:
    import pyimgano.models  # noqa: F401
    from pyimgano.models import create_model

    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 8)).astype(np.float32)

    det = create_model("core_knn_cosine", contamination=0.1, n_neighbors=5, method="largest")
    det.fit(X)
    scores = np.asarray(det.decision_function(X[:7]), dtype=np.float64).reshape(-1)
    preds = np.asarray(det.predict(X[:7]), dtype=int).reshape(-1)

    assert scores.shape == (7,)
    assert preds.shape == (7,)
    assert np.all(np.isfinite(scores))
    assert set(np.unique(preds)).issubset({0, 1})


def test_core_knn_cosine_accepts_torch_tensor_inputs() -> None:
    torch = __import__("torch")

    import pyimgano.models  # noqa: F401
    from pyimgano.models import create_model

    X = torch.randn(20, 6, dtype=torch.float32)
    det = create_model("core_knn_cosine", contamination=0.2, n_neighbors=3)
    det.fit(X)
    scores = np.asarray(det.decision_function(X), dtype=np.float64).reshape(-1)
    assert scores.shape == (20,)
    assert np.all(np.isfinite(scores))

