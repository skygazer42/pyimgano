from __future__ import annotations

import numpy as np
import pytest


def test_core_oddoneout_scores_outliers_higher() -> None:
    from pyimgano.models import create_model

    rng = np.random.default_rng(0)
    normal = rng.normal(0.0, 0.5, size=(64, 8)).astype(np.float32)
    outlier = rng.normal(4.0, 0.5, size=(8, 8)).astype(np.float32)

    x_train = normal
    x_test = np.concatenate([normal[:16], outlier], axis=0)

    det = create_model(
        "core_oddoneout",
        contamination=0.1,
        n_neighbors=5,
        metric="minkowski",
        p=2,
        normalize=True,
    )
    det.fit(x_train)
    scores = np.asarray(det.decision_function(x_test), dtype=np.float64).reshape(-1)

    assert scores.shape == (x_test.shape[0],)
    assert np.isfinite(scores).all()
    assert float(scores[-8:].mean()) > float(scores[:16].mean())


def test_core_oddoneout_accepts_torch_tensor_inputs() -> None:
    torch = pytest.importorskip("torch")

    from pyimgano.models import create_model

    x = torch.randn(32, 6)
    det = create_model("core_oddoneout", contamination=0.2, n_neighbors=3)
    det.fit(x)
    y = det.decision_function(x)

    assert hasattr(y, "shape")
