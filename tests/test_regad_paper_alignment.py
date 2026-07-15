from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")


def test_regad_paper_defaults_and_network_parameter_count() -> None:
    from pyimgano.models.regad import RegADModel, VisionRegAD

    detector = VisionRegAD(pretrained=False, device="cpu")
    assert detector.backbone == "resnet18"
    assert detector.image_size == 224
    assert detector.stn_mode == "rotation_scale"
    assert detector.learning_rate == pytest.approx(1e-4)
    assert detector.momentum == pytest.approx(0.9)
    assert detector.batch_size == 32
    assert detector.epochs == 50
    assert detector.shot == 2
    assert detector.covariance_regularization == pytest.approx(0.01)
    assert detector.gaussian_sigma == pytest.approx(4.0)

    model = RegADModel(pretrained=False)
    assert sum(parameter.numel() for parameter in model.parameters()) == 7_687_209
    assert model.registration.stn1.regressor[0].in_features == 3136
    assert model.registration.stn2.regressor[0].in_features == 784
    assert model.registration.stn3.regressor[0].in_features == 256
    torch.testing.assert_close(
        model.registration.stn1.regressor[2].bias,
        torch.tensor([0.0, 1.0, 1.0]),
    )


def test_regad_paper_preprocessing_and_support_augmentation() -> None:
    from pyimgano.models.regad import VisionRegAD

    detector = VisionRegAD(pretrained=False, image_size=32, device="cpu")
    images = np.stack(
        (
            np.zeros((32, 32, 3), dtype=np.uint8),
            np.full((32, 32, 3), 255, dtype=np.uint8),
        )
    )
    tensor = detector._preprocess(images)

    assert tensor.min().item() == pytest.approx(0.0)
    assert tensor.max().item() == pytest.approx(1.0)
    assert detector._augment_support(tensor).shape == (44, 3, 32, 32)


def test_regad_dual_gaussian_is_exact_mahalanobis() -> None:
    from pyimgano.models.regad import VisionRegAD

    torch.manual_seed(0)
    detector = VisionRegAD(
        pretrained=False,
        covariance_regularization=0.01,
        device="cpu",
    )
    support = torch.randn(5, 4, 2, 2)
    query = torch.randn(3, 4, 2, 2)
    detector._fit_gaussian(support)
    actual = detector._mahalanobis_map(query)

    expected = torch.empty_like(actual)
    for row in range(2):
        for column in range(2):
            samples = support[:, :, row, column]
            mean = samples.mean(dim=0)
            covariance = torch.cov(samples.T) + 0.01 * torch.eye(4)
            delta = query[:, :, row, column] - mean
            expected[:, row, column] = torch.sqrt(
                torch.einsum(
                    "bi,ij,bj->b",
                    delta,
                    torch.linalg.inv(covariance),
                    delta,
                )
            )

    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


def test_regad_refuses_same_category_proxy_inputs() -> None:
    from pyimgano.models.regad import VisionRegAD

    detector = VisionRegAD(pretrained=False, image_size=32, epochs=1, device="cpu")
    source = np.zeros((4, 32, 32, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="support_images"):
        detector.fit(source, np.asarray([0, 0, 1, 1]))

    detector = VisionRegAD(pretrained=False, image_size=32, epochs=0, device="cpu")
    with pytest.raises(ValueError, match="preloaded"):
        detector.fit(source, support_images=source[:2])
