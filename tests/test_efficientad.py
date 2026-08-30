from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from pyimgano.models.efficientad import (  # noqa: E402
    EfficientADAutoEncoder,
    EfficientADDetector,
    EfficientADModel,
    MediumPatchDescriptionNetwork,
    SmallPatchDescriptionNetwork,
    _hard_feature_loss,
    _normalize_map,
    _resize_anomaly_map,
)


class _ConstantMap(torch.nn.Module):
    def __init__(self, channels: int, value: float) -> None:
        super().__init__()
        self.channels = int(channels)
        self.register_buffer("value", torch.tensor(float(value)))

    def forward(self, images):  # noqa: ANN001, ANN201
        return self.value.expand(images.shape[0], self.channels, 2, 2)


def test_efficientad_paper_defaults_and_layer_tables() -> None:
    detector = EfficientADDetector(device="cpu")
    assert detector.image_size == (256, 256)
    assert detector.model_size == "small"
    assert detector.teacher_out_channels == 384
    assert detector.training_steps == 70_000
    assert detector.batch_size == 1
    assert detector.lr == pytest.approx(1e-4)
    assert detector.weight_decay == pytest.approx(1e-5)
    assert detector.padding is False
    assert detector.paper_strict is True

    small = SmallPatchDescriptionNetwork(384, padding=True)
    assert (small.conv1.in_channels, small.conv1.out_channels) == (3, 128)
    assert small.conv1.kernel_size == (4, 4)
    assert small.conv1.padding == (3, 3)
    assert small.pool1.kernel_size == 2
    assert small.pool1.padding == 1
    assert (small.conv2.in_channels, small.conv2.out_channels) == (128, 256)
    assert small.conv3.kernel_size == (3, 3)
    assert small.conv4.out_channels == 384

    medium = MediumPatchDescriptionNetwork(384, padding=True)
    assert [
        medium.conv1.out_channels,
        medium.conv2.out_channels,
        medium.conv3.out_channels,
        medium.conv4.out_channels,
        medium.conv5.out_channels,
        medium.conv6.out_channels,
    ] == [256, 512, 512, 512, 384, 384]
    assert [
        medium.conv1.kernel_size,
        medium.conv2.kernel_size,
        medium.conv3.kernel_size,
        medium.conv4.kernel_size,
        medium.conv5.kernel_size,
        medium.conv6.kernel_size,
    ] == [(4, 4), (4, 4), (1, 1), (3, 3), (4, 4), (1, 1)]

    autoencoder = EfficientADAutoEncoder(384, padding=True)
    assert len(autoencoder.encoder) == 6
    assert autoencoder.encoder[-1].kernel_size == (8, 8)
    assert autoencoder.encoder[-1].out_channels == 64
    assert len(autoencoder.decoder) == 8
    assert autoencoder.decoder[-1].out_channels == 384
    assert all(dropout.p == pytest.approx(0.2) for dropout in autoencoder.dropouts)

    small_student = SmallPatchDescriptionNetwork(768, padding=False)
    medium_student = MediumPatchDescriptionNetwork(768, padding=False)
    small_parameters = sum(
        parameter.numel()
        for module in (small, small_student, autoencoder)
        for parameter in module.parameters()
    )
    medium_parameters = sum(
        parameter.numel()
        for module in (medium, medium_student, autoencoder)
        for parameter in module.parameters()
    )
    assert small_parameters == 8_057_856  # paper efficiency table: 8M
    assert medium_parameters == 20_738_432  # paper efficiency table: 21M


@pytest.mark.parametrize(("padding", "side"), [(False, 56), (True, 64)])
def test_efficientad_pdn_student_and_autoencoder_shapes(padding: bool, side: int) -> None:
    model = EfficientADModel(model_size="small", teacher_out_channels=8, padding=padding).eval()
    images = torch.rand(1, 3, 256, 256)
    with torch.inference_mode():
        teacher = model.teacher(images)
        student = model.student(images)
        autoencoder = model.autoencoder(images)
    assert teacher.shape == (1, 8, side, side)
    assert student.shape == (1, 16, side, side)
    assert autoencoder.shape == teacher.shape


def test_efficientad_paper_losses_and_map_equations() -> None:
    model = EfficientADModel(teacher_out_channels=2, padding=False)
    model.teacher = _ConstantMap(2, 0.0)
    model.student = _ConstantMap(4, 1.0)
    model.autoencoder = _ConstantMap(2, 2.0)
    images = torch.rand(1, 3, 256, 256)

    loss_st, loss_ae, loss_stae = model.loss_terms(images, images)
    assert float(loss_st) == pytest.approx(2.0)
    assert float(loss_ae) == pytest.approx(4.0)
    assert float(loss_stae) == pytest.approx(1.0)

    model.qa_st.fill_(0.0)
    model.qb_st.fill_(1.0)
    model.qa_ae.fill_(0.0)
    model.qb_ae.fill_(1.0)
    with torch.inference_mode():
        anomaly_map = model.anomaly_map(images)
    assert anomaly_map.shape == (1, 1, 256, 256)
    torch.testing.assert_close(anomaly_map, torch.full_like(anomaly_map, 0.1))


def test_efficientad_hard_mining_and_quantile_normalization() -> None:
    distance = torch.tensor([0.0, 1.0, 2.0, 3.0])
    assert float(_hard_feature_loss(distance)) == pytest.approx(3.0)
    normalized = _normalize_map(torch.tensor([1.0, 3.0]), torch.tensor(1.0), torch.tensor(3.0))
    torch.testing.assert_close(normalized, torch.tensor([0.0, 0.1]))


def test_efficientad_restores_anomaly_map_to_original_image_size() -> None:
    source = np.arange(16, dtype=np.float32).reshape(4, 4)
    restored = _resize_anomaly_map(source, (7, 11))

    assert restored.shape == (7, 11)
    assert restored.dtype == np.float32
    assert np.isfinite(restored).all()


def test_efficientad_map_api_preserves_size_and_rejects_heterogeneous_stack() -> None:
    class _MapModel:
        def eval(self):  # noqa: ANN201
            return self

        def anomaly_map(self, images):  # noqa: ANN001, ANN201
            return torch.ones((images.shape[0], 1, 256, 256), dtype=torch.float32)

    detector = EfficientADDetector.__new__(EfficientADDetector)
    detector.model = _MapModel()
    detector.device = torch.device("cpu")
    detector.image_size = (256, 256)
    detector.batch_size = 2
    detector.eval_transform = None
    detector._check_is_fitted = lambda: None
    detector._loader = lambda values, **_kwargs: [
        (torch.zeros((len(values), 3, 256, 256)), torch.zeros(len(values)))
    ]

    same_size = [np.zeros((12, 20, 3), dtype=np.uint8) for _ in range(2)]
    assert detector.predict_anomaly_map(same_size).shape == (2, 12, 20)

    mixed_sizes = [
        np.zeros((12, 20, 3), dtype=np.uint8),
        np.zeros((9, 20, 3), dtype=np.uint8),
    ]
    with pytest.raises(ValueError, match="Inconsistent original image sizes"):
        detector.predict_anomaly_map(mixed_sizes)


def test_efficientad_strict_fit_requires_paper_assets() -> None:
    detector = EfficientADDetector(training_steps=1, teacher_out_channels=4, device="cpu")
    images = [
        np.zeros((32, 32, 3), dtype=np.uint8),
        np.ones((32, 32, 3), dtype=np.uint8),
    ]
    with pytest.raises(ValueError, match="teacher_checkpoint and imagenet_dir"):
        detector.fit(images)


def test_efficientad_teacher_and_detector_checkpoint_roundtrip(tmp_path) -> None:
    teacher = SmallPatchDescriptionNetwork(4, padding=False)
    teacher_path = tmp_path / "teacher.pth"
    torch.save(teacher.state_dict(), teacher_path)

    detector = EfficientADDetector(
        training_steps=1,
        teacher_out_channels=4,
        teacher_checkpoint=teacher_path,
        paper_strict=False,
        device="cpu",
    )
    model = detector.build_model()
    assert detector.teacher_checkpoint_loaded_ is True
    torch.testing.assert_close(model.teacher.conv1.weight, teacher.conv1.weight)

    model.teacher_mean.fill_(2.0)
    model.teacher_std.fill_(3.0)
    model.qa_st.fill_(0.1)
    model.qb_st.fill_(0.2)
    model.qa_ae.fill_(0.3)
    model.qb_ae.fill_(0.4)
    detector.threshold_ = 0.5
    detector.decision_scores_ = np.array([0.1, 0.9], dtype=np.float64)
    checkpoint = tmp_path / "efficientad.ckpt"
    detector.save_checkpoint(checkpoint)

    restored = EfficientADDetector(
        training_steps=1,
        teacher_out_channels=4,
        paper_strict=False,
        device="cpu",
        checkpoint_path=checkpoint,
    )
    assert restored.is_fitted_ is True
    assert restored.threshold_ == pytest.approx(0.5)
    np.testing.assert_allclose(restored.decision_scores_, [0.1, 0.9])
    torch.testing.assert_close(restored.model.teacher_mean, model.teacher_mean)
    torch.testing.assert_close(restored.model.qa_ae, model.qa_ae)
