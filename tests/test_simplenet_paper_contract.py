import inspect

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")


def test_simplenet_defaults_match_paper_training_contract() -> None:
    from pyimgano.models.simplenet import VisionSimpleNet

    parameters = inspect.signature(VisionSimpleNet).parameters
    assert parameters["backbone"].default == "wide_resnet50_2"
    assert parameters["feature_dim"].default == 1536
    assert parameters["discriminator_hidden_dim"].default == 1024
    assert parameters["patch_size"].default == 3
    assert parameters["patch_stride"].default == 1
    assert parameters["noise_std"].default == pytest.approx(0.015)
    assert parameters["discriminator_margin"].default == pytest.approx(0.5)
    assert parameters["lr"].default == pytest.approx(1e-4)
    assert parameters["discriminator_lr"].default == pytest.approx(2e-4)
    assert parameters["weight_decay"].default == pytest.approx(1e-5)
    assert parameters["epochs"].default == 160
    assert parameters["batch_size"].default == 4
    assert parameters["resize_size"].default == 256
    assert parameters["image_size"].default == 224
    assert parameters["gaussian_sigma"].default == pytest.approx(4.0)
    # Repository guardrail: exact ImageNet weights remain an explicit opt-in.
    assert parameters["pretrained"].default is False


def test_simplenet_adapter_and_discriminator_match_paper_modules() -> None:
    from pyimgano.models.simplenet import AnomalyDiscriminator, SimpleAdapter

    adapter = SimpleAdapter()
    assert adapter.projection.in_features == 1536
    assert adapter.projection.out_features == 1536
    assert adapter.projection.bias is None
    assert sum(parameter.numel() for parameter in adapter.parameters()) == 2_359_296

    discriminator = AnomalyDiscriminator()
    assert isinstance(discriminator.network[0], torch.nn.Linear)
    assert isinstance(discriminator.network[1], torch.nn.BatchNorm1d)
    assert isinstance(discriminator.network[2], torch.nn.LeakyReLU)
    assert discriminator.network[2].negative_slope == pytest.approx(0.2)
    assert isinstance(discriminator.network[3], torch.nn.Linear)
    assert discriminator.network[3].bias is None
    assert sum(parameter.numel() for parameter in discriminator.parameters()) == 1_576_960


def test_simplenet_patchify_uses_padded_three_by_three_neighborhoods() -> None:
    from pyimgano.models.simplenet import VisionSimpleNet

    feature_map = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    patches, grid = VisionSimpleNet._patchify(
        feature_map, patch_size=3, patch_stride=1
    )

    assert grid == (2, 2)
    assert patches.shape == (1, 4, 1, 3, 3)
    torch.testing.assert_close(
        patches[0, 0, 0],
        torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 2.0], [0.0, 3.0, 4.0]]),
    )


def test_simplenet_multilevel_embedding_uses_largest_patch_grid() -> None:
    from pyimgano.models.simplenet import VisionSimpleNet

    detector = VisionSimpleNet.__new__(VisionSimpleNet)
    detector.patch_size = 3
    detector.patch_stride = 1
    detector.feature_dim = 12
    feature_maps = [
        torch.arange(2 * 4 * 4, dtype=torch.float32).reshape(2, 1, 4, 4),
        torch.arange(2 * 2 * 2, dtype=torch.float32).reshape(2, 1, 2, 2),
    ]

    embedded, grid = detector._embed_feature_maps(feature_maps)

    assert grid == (4, 4)
    assert embedded.shape == (2 * 4 * 4, 12)
    assert torch.isfinite(embedded).all()
