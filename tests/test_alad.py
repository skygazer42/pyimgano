from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")


def test_alad_matches_paper_image_network_defaults_and_score() -> None:
    import torch

    from pyimgano.models.alad import ALAD, ConvDecoder, ConvEncoder, DiscXX, DiscXZ, DiscZZ
    from pyimgano.models.registry import MODEL_REGISTRY

    det = ALAD(device="cpu", verbose=0)
    model = det.build_model()

    assert (det.latent_dim, det.dropout_rate, det.batch_size, det.epoch_num) == (100, 0.2, 32, 100)
    assert (det.learning_rate_gen, det.learning_rate_disc) == (2e-4, 2e-4)
    assert (det.add_disc_zz_loss, det.spectral_normalization, det.score_degree) == (
        True,
        True,
        1.0,
    )
    assert (det.ema_enabled, det.ema_decay, det.ema_start_epoch) == (True, 0.999, 1)
    assert (det.validation_fraction, det.validation_patience, det.restore_best_validation) == (
        0.1,
        10,
        True,
    )
    assert [type(layer).__name__ for layer in det.train_transform.transforms] == [
        "Resize",
        "ToTensor",
        "Normalize",
    ]
    assert det.train_transform.transforms[0].size == (32, 32)
    assert tuple(det.train_transform.transforms[2].mean) == (0.5, 0.5, 0.5)
    assert tuple(det.train_transform.transforms[2].std) == (0.5, 0.5, 0.5)

    assert isinstance(det.enc, ConvEncoder)
    assert isinstance(det.dec, ConvDecoder)
    assert isinstance(det.disc_xz, DiscXZ)
    assert isinstance(det.disc_xx, DiscXX)
    assert isinstance(det.disc_zz, DiscZZ)
    assert [det.enc.conv1.out_channels, det.enc.conv2.out_channels, det.enc.conv3.out_channels] == [
        128,
        256,
        512,
    ]
    assert det.enc.conv4.out_channels == 100
    assert [
        det.dec.deconv1.out_channels,
        det.dec.deconv2.out_channels,
        det.dec.deconv3.out_channels,
        det.dec.deconv4.out_channels,
    ] == [512, 256, 128, 3]
    assert (det.disc_xx.conv1.in_channels, det.disc_xx.conv1.out_channels) == (6, 64)
    assert (det.disc_xx.conv2.in_channels, det.disc_xx.conv2.out_channels) == (64, 128)
    assert det.disc_zz.hidden[0].in_features == 200
    assert [sum(parameter.numel() for parameter in module.parameters()) for module in model] == [
        3_449_572,
        3_449_475,
        11_859_329,
        222_785,
        14_977,
    ]
    assert all(
        hasattr(layer, "weight_orig") for layer in (det.enc.conv1, det.enc.conv2, det.enc.conv3)
    )
    assert not hasattr(det.enc.conv4, "weight_orig")
    assert not hasattr(det.dec.deconv1, "weight_orig")
    assert det.enc.bn1.eps == pytest.approx(1e-3)
    assert det.enc.bn1.momentum == pytest.approx(0.01)

    for optimizer in (det.opt_gen, det.opt_enc, det.opt_disc):
        assert optimizer.defaults["lr"] == pytest.approx(2e-4)
        assert optimizer.defaults["betas"] == pytest.approx((0.5, 0.999))

    model.eval()
    images = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        reconstructed = det.dec(det.enc(images))
        expected = torch.norm(
            det.disc_xx.features(images, images) - det.disc_xx.features(images, reconstructed),
            p=1,
            dim=1,
        ).numpy()
    actual = det.evaluating_forward((images, torch.zeros(2)))
    assert actual == pytest.approx(expected)
    assert MODEL_REGISTRY.info("vision_alad").metadata["paper_fidelity"] == "core-aligned"


def test_vision_alad_contract_fit_and_score() -> None:
    from pyimgano.models import create_model

    rng = np.random.default_rng(10)
    train = [rng.integers(0, 255, size=(32, 32, 3), dtype=np.uint8) for _ in range(4)]
    test = [rng.integers(0, 255, size=(32, 32, 3), dtype=np.uint8) for _ in range(2)]

    det = create_model(
        "vision_alad",
        preprocessing=False,
        epoch_num=1,
        batch_size=2,
        device="cpu",
        verbose=0,
    )

    det.fit(train)
    assert det.training_ema_updates_ == 2
    assert det.training_ema_applied_ is True
    discriminator_parameters = {
        id(parameter) for group in det.opt_disc.param_groups for parameter in group["params"]
    }
    assert all(id(parameter) in discriminator_parameters for parameter in det.img_feat.parameters())
    scores = np.asarray(det.decision_function(test), dtype=np.float64).reshape(-1)
    assert scores.shape == (2,)
    assert np.all(np.isfinite(scores))
    assert len(det.validation_score_history_) == 1
    assert det.best_validation_epoch_ == 1


def test_alad_validation_selection_stops_and_restores_best_state() -> None:
    import torch

    from pyimgano.models.alad import ALAD

    detector = ALAD(
        device="cpu",
        verbose=0,
        validation_patience=2,
        restore_best_validation=True,
    )
    detector.model = torch.nn.Linear(2, 1, bias=False)
    detector._validation_loader = object()
    score_sequence = iter([np.asarray([0.2, 0.4]), np.asarray([0.5]), np.asarray([0.6])])
    detector.evaluate = lambda _loader: next(score_sequence)  # type: ignore[method-assign]

    with torch.no_grad():
        detector.model.weight.fill_(1.0)
    detector.training_epochs_completed_ = 1
    detector.epoch_update()

    with torch.no_grad():
        detector.model.weight.fill_(2.0)
    detector.training_epochs_completed_ = 2
    detector.epoch_update()
    detector.training_epochs_completed_ = 3
    detector.epoch_update()

    assert detector.training_stop_reason_ == "validation_early_stopping"
    assert detector.best_validation_score_ == pytest.approx(0.3)
    assert detector.best_validation_epoch_ == 1
    assert detector._restore_best_validation_state() is True
    assert torch.equal(detector.model.weight, torch.ones_like(detector.model.weight))
