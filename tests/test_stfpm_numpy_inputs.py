import numpy as np
import pytest

from pyimgano.models import create_model

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")


def test_stfpm_accepts_numpy_images_for_scoring_and_maps():
    det = create_model(
        "vision_stfpm",
        pretrained_teacher=False,
        epochs=1,
        batch_size=1,
        device="cpu",
    )

    # Avoid running full training in this unit test; enable scoring path.
    det._is_fitted = True

    imgs = [np.zeros((32, 48, 3), dtype=np.uint8) for _ in range(2)]
    scores = det.decision_function(imgs)
    assert scores.shape == (2,)

    anomaly_map = det.get_anomaly_map(imgs[0])
    assert anomaly_map.ndim == 2
    assert anomaly_map.shape == (32, 48)
    assert np.isfinite(anomaly_map).all()
    assert scores[0] == pytest.approx(float(anomaly_map.max()))


def test_stfpm_uses_paper_train_validation_split() -> None:
    det = create_model(
        "vision_stfpm",
        pretrained_teacher=False,
        epochs=1,
        batch_size=2,
        validation_ratio=0.2,
        device="cpu",
    )
    inputs = [f"image-{index}.png" for index in range(12)]

    train, validation = det._split_training_inputs(inputs)

    assert det.random_state == 0
    assert len(train) == 9
    assert validation == ["image-6.png", "image-11.png", "image-4.png"]
    assert set(train).isdisjoint(validation)
    assert sorted(train + validation) == sorted(inputs)


def test_stfpm_paper_network_loss_and_anomaly_map_contract() -> None:
    det = create_model(
        "vision_stfpm",
        pretrained_teacher=False,
        device="cpu",
    )

    assert det.layers == ["layer1", "layer2", "layer3"]
    assert det.epochs == 100
    assert det.batch_size == 32
    assert det.lr == pytest.approx(0.4)
    assert all(not parameter.requires_grad for parameter in det.teacher.parameters())
    assert sum(parameter.numel() for parameter in det.teacher.parameters()) == sum(
        parameter.numel() for parameter in det.student.parameters()
    )

    with torch.no_grad():
        features = det._extract_features(det.teacher, torch.zeros(1, 3, 256, 256))
    assert {name: tuple(value.shape) for name, value in features.items()} == {
        "layer1": (1, 64, 64, 64),
        "layer2": (1, 128, 32, 32),
        "layer3": (1, 256, 16, 16),
    }

    teacher = torch.tensor([[[[1.0]], [[0.0]]]])
    student = torch.tensor([[[[1.0]], [[1.0]]]])
    teacher_features = {layer: teacher for layer in det.layers}
    student_features = {layer: student for layer in det.layers}
    per_level = 1.0 - 2.0**-0.5

    loss = det._feature_matching_loss(teacher_features, student_features)
    anomaly_map = det._feature_anomaly_map(
        teacher_features,
        student_features,
        output_size=(5, 7),
    )
    assert float(loss) == pytest.approx(3.0 * per_level)
    assert anomaly_map.shape == (1, 1, 5, 7)
    np.testing.assert_allclose(anomaly_map.numpy(), per_level**3, rtol=1e-6)
