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


def test_promptad_build_feature_extractor_uses_shared_torchvision_loader(monkeypatch) -> None:
    import torch

    import pyimgano.models.promptad as promptad_module
    from pyimgano.models.promptad import VisionPromptAD

    class _FakeResNet:
        def __init__(self) -> None:
            self.conv1 = torch.nn.Identity()
            self.bn1 = torch.nn.Identity()
            self.relu = torch.nn.Identity()
            self.maxpool = torch.nn.Identity()
            self.layer1 = torch.nn.Identity()
            self.layer2 = torch.nn.Identity()
            self.layer3 = torch.nn.Identity()

    calls: list[tuple[str, bool]] = []

    def _fake_loader(name: str, *, pretrained: bool):
        calls.append((name, pretrained))
        return _FakeResNet(), None

    monkeypatch.setattr(promptad_module, "load_torchvision_model", _fake_loader, raising=False)

    det = VisionPromptAD(backbone="resnet18", device="cpu")
    extractor = det._build_feature_extractor()

    assert isinstance(extractor, torch.nn.Sequential)
    assert calls == [("resnet18", True)]
