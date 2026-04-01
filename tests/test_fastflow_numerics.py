from __future__ import annotations

import pytest


def test_fastflow_invconv_logdet_is_finite_for_negative_determinant_seed() -> None:
    torch = pytest.importorskip("torch")

    from pyimgano.models.fastflow import InvConv2d

    torch.manual_seed(0)
    layer = InvConv2d(4)

    value = layer._log_det()
    assert torch.isfinite(value).item() is True


def test_fastflow_invconv_forward_produces_finite_logdet() -> None:
    torch = pytest.importorskip("torch")

    from pyimgano.models.fastflow import InvConv2d

    torch.manual_seed(0)
    layer = InvConv2d(4)
    x = torch.randn(2, 4, 8, 8)
    logdet = torch.zeros(2)

    _y, updated = layer(x, logdet, reverse=False)
    assert torch.isfinite(updated).all().item() is True


def test_fastflow_feature_extractor_uses_shared_torchvision_loader(monkeypatch) -> None:
    torch = pytest.importorskip("torch")

    import pyimgano.models.fastflow as fastflow_module
    from pyimgano.models.fastflow import ResNetFeatureExtractor

    class _FakeResNet:
        def __init__(self) -> None:
            self.conv1 = torch.nn.Identity()
            self.bn1 = torch.nn.Identity()
            self.relu = torch.nn.Identity()
            self.maxpool = torch.nn.Identity()
            self.layer1 = torch.nn.Identity()
            self.layer2 = torch.nn.Identity()
            self.layer3 = torch.nn.Identity()
            self.layer4 = torch.nn.Identity()

    calls: list[tuple[str, bool]] = []

    def _fake_loader(name: str, *, pretrained: bool):
        calls.append((name, pretrained))
        return _FakeResNet(), None

    monkeypatch.setattr(fastflow_module, "load_torchvision_model", _fake_loader, raising=False)

    extractor = ResNetFeatureExtractor(backbone="resnet18", pretrained=False)
    outputs = extractor(torch.zeros((1, 3, 8, 8), dtype=torch.float32))

    assert len(outputs) == 3
    assert calls == [("resnet18", False)]
