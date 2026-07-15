import inspect

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")


def test_reverse_distillation_uses_bottleneck_and_reverse_decoder() -> None:
    from pyimgano.models.reverse_distillation import ReverseDistillation

    detector = ReverseDistillation(
        pretrained_backbone=False,
        epoch_num=1,
        batch_size=1,
        device="cpu",
        verbose=0,
    )
    detector.model = detector.build_model()

    images = torch.zeros((1, 3, 64, 64), dtype=torch.float32)
    teacher, student = detector._forward_features(images)

    assert [tuple(item.shape) for item in teacher] == [
        (1, 256, 16, 16),
        (1, 512, 8, 8),
        (1, 1024, 4, 4),
    ]
    assert [tuple(item.shape) for item in student] == [
        (1, 256, 16, 16),
        (1, 512, 8, 8),
        (1, 1024, 4, 4),
    ]
    assert len(detector.bottleneck.bn_layer) == 3
    assert [len(detector.decoder.layer1), len(detector.decoder.layer2), len(detector.decoder.layer3)] == [
        3,
        4,
        6,
    ]
    assert isinstance(detector.decoder.layer1[0].conv2, torch.nn.ConvTranspose2d)
    assert sum(parameter.numel() for parameter in detector.teacher.parameters()) == 68_883_240
    assert sum(parameter.numel() for parameter in detector.bottleneck.parameters()) == 67_277_824
    assert sum(parameter.numel() for parameter in detector.decoder.parameters()) == 24_917_504


def test_reverse_distillation_defaults_match_author_training_path() -> None:
    from pyimgano.models.reverse_distillation import ReverseDistillation

    parameters = inspect.signature(ReverseDistillation).parameters
    assert parameters["backbone"].default == "wide_resnet50_2"
    assert parameters["pretrained_backbone"].default is True
    assert parameters["image_size"].default == 256
    assert parameters["epoch_num"].default == 200
    assert parameters["batch_size"].default == 16
    assert parameters["lr"].default == pytest.approx(5e-3)
    assert parameters["anomaly_smoothing_sigma"].default == pytest.approx(4.0)


def test_reverse_distillation_rejects_forward_distillation_layers() -> None:
    from pyimgano.models.reverse_distillation import ReverseDistillation

    with pytest.raises(ValueError, match="layer1.*layer2.*layer3"):
        ReverseDistillation(
            selected_layers=("layer2", "layer3", "layer4"),
            pretrained_backbone=False,
            device="cpu",
        )


def test_reverse_distillation_anomaly_map_is_channel_cosine_distance() -> None:
    from pyimgano.models.reverse_distillation import ReverseDistillation

    detector = ReverseDistillation(
        pretrained_backbone=False,
        anomaly_map_mode="add",
        device="cpu",
        verbose=0,
    )
    teacher = [torch.tensor([[[[1.0]], [[0.0]]]]) for _ in range(3)]
    student = [torch.tensor([[[[0.0]], [[1.0]]]]) for _ in range(3)]

    anomaly_map = detector._anomaly_maps(teacher, student, output_size=(2, 2))

    np.testing.assert_allclose(anomaly_map.numpy(), np.full((1, 1, 2, 2), 3.0))


def test_reverse_distillation_checkpoint_includes_frozen_teacher(tmp_path, monkeypatch) -> None:
    import pyimgano.models.reverse_distillation as reverse_distillation

    class _TinyReverseDistillationNetwork(torch.nn.Module):
        def __init__(self, *, pretrained_backbone: bool) -> None:
            super().__init__()
            del pretrained_backbone
            self.teacher = torch.nn.Linear(1, 1, bias=False)
            self.teacher.weight.requires_grad_(False)
            self.bottleneck = torch.nn.Linear(1, 1)
            self.decoder = torch.nn.Linear(1, 1)

    monkeypatch.setattr(
        reverse_distillation,
        "ReverseDistillationNetwork",
        _TinyReverseDistillationNetwork,
    )
    ReverseDistillation = reverse_distillation.ReverseDistillation

    detector = ReverseDistillation(
        pretrained_backbone=False,
        epoch_num=1,
        batch_size=1,
        device="cpu",
        verbose=0,
        random_state=7,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        detector.model = detector.build_model()
    detector.threshold_ = 1.25
    expected = {
        key: value.detach().clone() for key, value in detector.model.state_dict().items()
    }
    assert any(key.startswith("teacher.") for key in expected)

    checkpoint = tmp_path / "reverse_distillation.ckpt"
    detector.save_checkpoint(checkpoint)

    restored = ReverseDistillation(
        pretrained_backbone=False,
        epoch_num=1,
        batch_size=1,
        device="cpu",
        verbose=0,
        random_state=99,
    )
    restored.load_checkpoint(checkpoint)

    actual = restored.model.state_dict()
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        assert torch.equal(actual[key], value), key
    assert restored.threshold_ == pytest.approx(1.25)
