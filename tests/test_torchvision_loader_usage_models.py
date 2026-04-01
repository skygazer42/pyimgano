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


def test_ast_teacher_encoder_uses_shared_torchvision_loader(monkeypatch) -> None:
    import torch

    import pyimgano.models.ast as ast_module
    from pyimgano.models.ast import TeacherEncoder

    calls: list[tuple[str, bool]] = []

    def _fake_loader(name: str, *, pretrained: bool):
        calls.append((name, pretrained))
        return _fake_resnet(torch), None

    monkeypatch.setattr(ast_module, "load_torchvision_model", _fake_loader, raising=False)

    encoder = TeacherEncoder(backbone="resnet18")
    out = encoder(torch.zeros((1, 3, 8, 8), dtype=torch.float32))

    assert tuple(out.shape) == (1, 3, 8, 8)
    assert calls == [("resnet18", True)]


def test_dst_teacher_network_uses_shared_torchvision_loader(monkeypatch) -> None:
    import torch

    import pyimgano.models.dst as dst_module
    from pyimgano.models.dst import TeacherNetwork

    calls: list[tuple[str, bool]] = []

    def _fake_loader(name: str, *, pretrained: bool):
        calls.append((name, pretrained))
        return _fake_resnet(torch), None

    monkeypatch.setattr(dst_module, "load_torchvision_model", _fake_loader, raising=False)

    network = TeacherNetwork(backbone="resnet18")
    outputs = network(torch.zeros((1, 3, 8, 8), dtype=torch.float32))

    assert len(outputs) == 3
    assert calls == [("resnet18", True)]


def test_favae_feature_extractor_uses_shared_torchvision_loader(monkeypatch) -> None:
    import torch

    import pyimgano.models.favae as favae_module
    from pyimgano.models.favae import FeatureExtractor

    calls: list[tuple[str, bool]] = []

    def _fake_loader(name: str, *, pretrained: bool):
        calls.append((name, pretrained))
        return _fake_resnet(torch), None

    monkeypatch.setattr(favae_module, "load_torchvision_model", _fake_loader, raising=False)

    extractor = FeatureExtractor(backbone="resnet18")
    out = extractor(torch.zeros((1, 3, 8, 8), dtype=torch.float32))

    assert tuple(out.shape) == (1, 3, 8, 8)
    assert calls == [("resnet18", True)]


def test_csflow_feature_extractor_uses_shared_torchvision_loader(monkeypatch) -> None:
    import torch

    import pyimgano.models.csflow as csflow_module
    from pyimgano.models.csflow import MultiScaleFeatureExtractor

    calls: list[tuple[str, bool]] = []

    def _fake_loader(name: str, *, pretrained: bool):
        calls.append((name, pretrained))
        return _fake_resnet(torch), None

    monkeypatch.setattr(csflow_module, "load_torchvision_model", _fake_loader, raising=False)

    extractor = MultiScaleFeatureExtractor(backbone="resnet18", pretrained=False)
    outputs = extractor(torch.zeros((1, 3, 8, 8), dtype=torch.float32))

    assert len(outputs) == 3
    assert calls == [("resnet18", False)]


def test_rdplusplus_encoder_uses_shared_torchvision_loader(monkeypatch) -> None:
    import torch

    import pyimgano.models.rdplusplus as rdplusplus_module
    from pyimgano.models.rdplusplus import MultiScaleEncoder

    calls: list[tuple[str, bool]] = []

    def _fake_loader(name: str, *, pretrained: bool):
        calls.append((name, pretrained))
        return _fake_resnet(torch), None

    monkeypatch.setattr(rdplusplus_module, "load_torchvision_model", _fake_loader, raising=False)

    encoder = MultiScaleEncoder(backbone="resnet18", pretrained=False)
    outputs = encoder(torch.zeros((1, 3, 8, 8), dtype=torch.float32))

    assert len(outputs) == 3
    assert calls == [("resnet18", False)]


def test_bayesianpf_feature_extractor_uses_shared_torchvision_loader(monkeypatch) -> None:
    import torch

    import pyimgano.models.bayesianpf as bayesianpf_module
    from pyimgano.models.bayesianpf import VisionBayesianPF

    calls: list[tuple[str, bool]] = []

    def _fake_loader(name: str, *, pretrained: bool):
        calls.append((name, pretrained))
        return _fake_resnet(torch), None

    monkeypatch.setattr(bayesianpf_module, "load_torchvision_model", _fake_loader, raising=False)

    det = VisionBayesianPF(backbone="resnet18", device="cpu", random_state=0)
    extractor = det._build_feature_extractor()

    assert isinstance(extractor, torch.nn.Sequential)
    assert calls == [("resnet18", True)]


def test_glad_feature_extractor_uses_shared_torchvision_loader(monkeypatch) -> None:
    import torch

    import pyimgano.models.glad as glad_module
    from pyimgano.models.glad import VisionGLAD

    calls: list[tuple[str, bool]] = []

    def _fake_loader(name: str, *, pretrained: bool):
        calls.append((name, pretrained))
        return _fake_resnet(torch), None

    monkeypatch.setattr(glad_module, "load_torchvision_model", _fake_loader, raising=False)

    det = VisionGLAD(backbone="resnet18", device="cpu", random_state=0)
    extractor = det._build_feature_extractor()

    assert isinstance(extractor, torch.nn.Sequential)
    assert calls == [("resnet18", True)]


def test_panda_encoder_uses_shared_torchvision_loader(monkeypatch) -> None:
    import torch

    import pyimgano.models.panda as panda_module
    from pyimgano.models.panda import PrototypicalEncoder

    calls: list[tuple[str, bool]] = []

    def _fake_loader(name: str, *, pretrained: bool):
        calls.append((name, pretrained))
        return _fake_resnet(torch), None

    monkeypatch.setattr(panda_module, "load_torchvision_model", _fake_loader, raising=False)

    encoder = PrototypicalEncoder(backbone="resnet18", projection_dim=64)
    out = encoder.backbone(torch.zeros((1, 3, 8, 8), dtype=torch.float32))

    assert tuple(out.shape) == (1, 3, 8, 8)
    assert calls == [("resnet18", True)]


def test_inctrl_encoder_uses_shared_torchvision_loader(monkeypatch) -> None:
    import torch

    import pyimgano.models.inctrl as inctrl_module
    from pyimgano.models.inctrl import ResidualEncoder

    calls: list[tuple[str, bool]] = []

    def _fake_loader(name: str, *, pretrained: bool):
        calls.append((name, pretrained))
        return _fake_resnet(torch), None

    monkeypatch.setattr(inctrl_module, "load_torchvision_model", _fake_loader, raising=False)

    encoder = ResidualEncoder(backbone="resnet18", feature_dim=64)
    out = encoder.backbone(torch.zeros((1, 3, 8, 8), dtype=torch.float32))

    assert tuple(out.shape) == (1, 3, 8, 8)
    assert calls == [("resnet18", True)]


def test_ast_teacher_encoder_uses_shared_torchvision_loader(monkeypatch) -> None:
    import torch

    import pyimgano.models.ast as ast_module
    from pyimgano.models.ast import TeacherEncoder

    calls: list[tuple[str, bool]] = []

    def _fake_loader(name: str, *, pretrained: bool):
        calls.append((name, pretrained))
        return _fake_resnet(torch), None

    monkeypatch.setattr(ast_module, "load_torchvision_model", _fake_loader, raising=False)

    encoder = TeacherEncoder(backbone="resnet18")
    out = encoder(torch.zeros((1, 3, 8, 8), dtype=torch.float32))

    assert tuple(out.shape) == (1, 3, 8, 8)
    assert calls == [("resnet18", True)]


def test_dst_teacher_network_uses_shared_torchvision_loader(monkeypatch) -> None:
    import torch

    import pyimgano.models.dst as dst_module
    from pyimgano.models.dst import TeacherNetwork

    calls: list[tuple[str, bool]] = []

    def _fake_loader(name: str, *, pretrained: bool):
        calls.append((name, pretrained))
        return _fake_resnet(torch), None

    monkeypatch.setattr(dst_module, "load_torchvision_model", _fake_loader, raising=False)

    network = TeacherNetwork(backbone="resnet18")
    outputs = network(torch.zeros((1, 3, 8, 8), dtype=torch.float32))

    assert len(outputs) == 3
    assert calls == [("resnet18", True)]
