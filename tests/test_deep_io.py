from __future__ import annotations

import numpy as np
import pytest


def test_safe_torch_load_accepts_numpy_array_payload(tmp_path) -> None:
    torch = pytest.importorskip("torch")

    from pyimgano.models.deep_io import safe_torch_load

    payload = {
        "memory_bank": np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "decision_scores_": np.asarray([0.1, 0.2], dtype=np.float64),
        "threshold_": 0.15,
    }
    ckpt = tmp_path / "payload.pt"
    torch.save(payload, ckpt)

    loaded = safe_torch_load(ckpt, map_location="cpu")

    assert isinstance(loaded, dict)
    np.testing.assert_allclose(loaded["memory_bank"], payload["memory_bank"])
    np.testing.assert_allclose(loaded["decision_scores_"], payload["decision_scores_"])
    assert float(loaded["threshold_"]) == pytest.approx(0.15)
