from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")


def test_promptad_fit_handles_feature_dim_larger_than_prompt_dim(monkeypatch) -> None:
    import torch

    from pyimgano.models.promptad import VisionPromptAD

    class _DummyExtractor(torch.nn.Module):
        def forward(self, x):  # noqa: ANN001
            batch = int(x.shape[0])
            return torch.ones((batch, 1024), dtype=torch.float32, device=x.device)

    monkeypatch.setattr(VisionPromptAD, "_build_feature_extractor", lambda self: _DummyExtractor())

    rng = np.random.default_rng(11)
    train = rng.integers(0, 255, size=(4, 32, 32, 3), dtype=np.uint8)
    test = rng.integers(0, 255, size=(2, 32, 32, 3), dtype=np.uint8)

    det = VisionPromptAD(
        backbone="wide_resnet50",
        epochs=1,
        batch_size=2,
        device="cpu",
        prompt_dim=512,
    )

    det.fit(train)
    scores = np.asarray(det.decision_function(test), dtype=np.float64).reshape(-1)
    assert scores.shape == (2,)
    assert np.all(np.isfinite(scores))
