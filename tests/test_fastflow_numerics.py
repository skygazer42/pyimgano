from __future__ import annotations

import inspect

import pytest


def test_fastflow_channel_permutation_is_fixed_and_invertible() -> None:
    torch = pytest.importorskip("torch")

    from pyimgano.models.fastflow import ChannelPermutation

    torch.manual_seed(0)
    layer = ChannelPermutation(8)
    x = torch.randn(2, 8, 5, 5)
    logdet = torch.randn(2, 5, 5)

    transformed, transformed_logdet = layer(x, logdet)
    restored, restored_logdet = layer(transformed, transformed_logdet, reverse=True)

    torch.testing.assert_close(restored, x)
    torch.testing.assert_close(restored_logdet, logdet)
    assert list(layer.parameters()) == []


def test_fastflow_affine_coupling_has_paper_subnet_and_spatial_logdet() -> None:
    torch = pytest.importorskip("torch")

    from pyimgano.models.fastflow import AffineCoupling

    torch.manual_seed(0)
    layer = AffineCoupling(64, hidden_ratio=1.0, kernel_size=3)
    x = torch.randn(2, 64, 8, 8)

    transformed, updated = layer(x)
    restored, reverse_logdet = layer(transformed, updated, reverse=True)

    convolutions = [module for module in layer.subnet if isinstance(module, torch.nn.Conv2d)]
    assert [module.kernel_size for module in convolutions] == [(3, 3), (3, 3)]
    assert convolutions[0].in_channels == 32
    assert convolutions[0].out_channels == 32
    assert convolutions[1].out_channels == 64
    assert updated.shape == (2, 8, 8)
    assert torch.isfinite(updated).all().item() is True
    torch.testing.assert_close(restored, x, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(reverse_logdet, torch.zeros_like(reverse_logdet), atol=1e-5, rtol=0)


def test_fastflow_resnet_flow_parameter_counts_match_paper_table() -> None:
    pytest.importorskip("torch")

    from pyimgano.models.fastflow import FlowStage

    def parameter_count(*, conv3x3_only: bool) -> int:
        stages = [
            FlowStage(
                channels,
                n_steps=8,
                hidden_ratio=1.0,
                conv3x3_only=conv3x3_only,
                affine_clamp=2.0,
            )
            for channels in (64, 128, 256)
        ]
        return sum(parameter.numel() for stage in stages for parameter in stage.parameters())

    # The paper reports rounded additional-parameter totals of 4.9 M (3-3)
    # and 2.7 M (alternating 3-1) for ResNet18.
    assert parameter_count(conv3x3_only=True) == 4_657_408
    assert parameter_count(conv3x3_only=False) == 2_593_024


def test_fastflow_paper_defaults_are_exposed() -> None:
    pytest.importorskip("torch")

    from pyimgano.models.fastflow import FastFlow

    parameters = inspect.signature(FastFlow).parameters
    assert parameters["selected_layers"].default == ("layer1", "layer2", "layer3")
    assert parameters["image_size"].default == 256
    assert parameters["n_flow_steps"].default == 8
    assert parameters["flow_hidden_ratio"].default == 1.0
    assert parameters["lr"].default == 1e-3
    assert parameters["weight_decay"].default == 1e-5
    assert parameters["epoch_num"].default == 500
    assert parameters["batch_size"].default == 32

    detector = FastFlow(image_size=32, verbose=0)
    assert detector.weight_decay == pytest.approx(1e-5)


def test_fastflow_uses_spatial_likelihood_and_channel_probability_map() -> None:
    torch = pytest.importorskip("torch")

    from pyimgano.models.fastflow import FastFlow

    latent = torch.tensor([[[[0.0]], [[2.0]]]])
    logdet = torch.tensor([[[0.5]]])

    # (0.5 * (0^2 + 2^2) - 0.5) / two latent dimensions.
    torch.testing.assert_close(FastFlow._flow_nll(latent, logdet), torch.tensor([0.75]))
    expected_probability = (1.0 + torch.exp(torch.tensor(-2.0))) / 2.0
    torch.testing.assert_close(
        FastFlow._latent_anomaly_map(latent),
        (1.0 - expected_probability).reshape(1, 1, 1),
    )


def test_fastflow_alternating_stage_starts_with_3x3() -> None:
    torch = pytest.importorskip("torch")

    from pyimgano.models.fastflow import FlowStage

    stage = FlowStage(
        8,
        n_steps=4,
        hidden_ratio=1.0,
        conv3x3_only=False,
        affine_clamp=2.0,
    )
    kernels = [
        step.coupling.subnet[0].kernel_size
        for step in stage.steps
        if isinstance(step.coupling.subnet[0], torch.nn.Conv2d)
    ]
    assert kernels == [(3, 3), (1, 1), (3, 3), (1, 1)]


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
