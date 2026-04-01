from __future__ import annotations

import pytest

pytest.importorskip("torch")


def _fake_resnet(torch):
    class _FakeResNet:
        def __init__(self) -> None:
            self.conv1 = torch.nn.Identity()
            self.bn1 = torch.nn.Identity()
            self.relu = torch.nn.Identity()
            self.maxpool = torch.nn.Identity()
            self.layer1 = torch.nn.Identity()
            self.layer2 = torch.nn.Identity()
            self.layer3 = torch.nn.Identity()

    return _FakeResNet()


def test_bgad_feature_extractor_uses_shared_torchvision_loader(monkeypatch) -> None:
    import torch

    import pyimgano.models.bgad as bgad_module
    from pyimgano.models.bgad import FeatureExtractor

    calls: list[tuple[str, bool]] = []

    def _fake_loader(name: str, *, pretrained: bool):
        calls.append((name, pretrained))
        return _fake_resnet(torch), None

    monkeypatch.setattr(bgad_module, "load_torchvision_model", _fake_loader, raising=False)

    extractor = FeatureExtractor(backbone="resnet18", pretrained=False)
    out = extractor(torch.zeros((1, 3, 8, 8), dtype=torch.float32))

    assert tuple(out.shape) == (1, 3, 8, 8)
    assert calls == [("resnet18", False)]


def test_dsr_feature_extractor_uses_shared_torchvision_loader(monkeypatch) -> None:
    import torch

    import pyimgano.models.dsr as dsr_module
    from pyimgano.models.dsr import FeatureExtractor

    calls: list[tuple[str, bool]] = []

    def _fake_loader(name: str, *, pretrained: bool):
        calls.append((name, pretrained))
        return _fake_resnet(torch), None

    monkeypatch.setattr(dsr_module, "load_torchvision_model", _fake_loader, raising=False)

    extractor = FeatureExtractor(backbone="resnet18", pretrained=False)
    out = extractor(torch.zeros((1, 3, 8, 8), dtype=torch.float32))

    assert tuple(out.shape) == (1, 3, 8, 8)
    assert calls == [("resnet18", False)]


def test_pni_feature_extractor_uses_shared_torchvision_loader(monkeypatch) -> None:
    import torch

    import pyimgano.models.pni as pni_module
    from pyimgano.models.pni import PyramidFeatureExtractor

    calls: list[tuple[str, bool]] = []

    def _fake_loader(name: str, *, pretrained: bool):
        calls.append((name, pretrained))
        return _fake_resnet(torch), None

    monkeypatch.setattr(pni_module, "load_torchvision_model", _fake_loader, raising=False)

    extractor = PyramidFeatureExtractor(
        backbone="resnet18",
        feature_levels=["layer1", "layer2", "layer3"],
        pretrained=False,
    )
    out = extractor(torch.zeros((1, 3, 8, 8), dtype=torch.float32))

    assert set(out.keys()) == {"layer1", "layer2", "layer3"}
    assert calls == [("resnet18", False)]
