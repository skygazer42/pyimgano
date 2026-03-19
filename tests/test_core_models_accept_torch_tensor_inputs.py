from __future__ import annotations

import numpy as np
import pytest


def test_core_models_accept_torch_tensor_inputs() -> None:
    torch = pytest.importorskip("torch")

    from pyimgano.models.registry import create_model

    x = torch.randn(32, 12, dtype=torch.float32)

    # Keep this fast: exercise a small representative set of core_* models.
    # The key contract is that `CoreFeatureDetector` can accept torch tensors
    # and converts them to a 2D NumPy feature matrix.
    models = [
        ("core_iforest", {"n_estimators": 10, "random_state": 0}),
        ("core_lof", {"n_neighbors": 5}),
    ]

    for name, extra_kwargs in models:
        det = create_model(name, contamination=0.1, **dict(extra_kwargs))
        det.fit(x)
        scores = det.decision_function(x)

        scores_arr = np.asarray(scores, dtype=np.float64).reshape(-1)
        assert scores_arr.shape == (x.shape[0],)
        assert np.all(np.isfinite(scores_arr))
