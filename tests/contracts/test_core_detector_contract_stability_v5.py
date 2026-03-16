from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.parametrize(
    "model_name",
    [
        # Keep this short: representative set + v5 additions.
        "core_knn_cosine",
        "core_knn_cosine_calibrated",
        "core_mahalanobis_shrinkage",
        "core_cosine_mahalanobis",
        "core_oddoneout",
    ],
)
def test_core_detector_contract_torch_inputs_and_nan_handling(model_name: str) -> None:
    torch = pytest.importorskip("torch")

    import pyimgano.models  # noqa: F401 - registry population side effects
    from pyimgano.models.registry import create_model

    rng = np.random.default_rng(0)
    x_np = rng.standard_normal(size=(96, 12)).astype(np.float32)
    x_torch = torch.as_tensor(x_np, dtype=torch.float32)

    det = create_model(model_name, contamination=0.1)
    det.fit(x_torch)

    scores = np.asarray(det.decision_function(x_torch[:11]), dtype=np.float64).reshape(-1)
    assert scores.shape == (11,)
    assert np.all(np.isfinite(scores))

    x_bad = x_torch.clone()
    x_bad[0, 0] = float("nan")

    try:
        out = det.decision_function(x_bad[:7])
    except ValueError:
        return

    out_arr = np.asarray(out, dtype=np.float64).reshape(-1)
    assert out_arr.shape == (7,)
    assert np.all(np.isfinite(out_arr))
