import numpy as np
import pytest

from pyimgano.models import create_model

pytest.importorskip("torch")
pytest.importorskip("torchvision")


def test_draem_accepts_numpy_images_for_scoring_and_maps():
    det = create_model(
        "vision_draem",
        image_size=32,
        epochs=1,
        batch_size=1,
        device="cpu",
        base_channels=4,
    )

    # Avoid running full training in this unit test; enable inference path.
    det._is_fitted = True

    imgs = [np.zeros((20, 30, 3), dtype=np.uint8) for _ in range(2)]
    scores = det.decision_function(imgs)
    assert scores.shape == (2,)

    anomaly_map = det.get_anomaly_map(imgs[0])
    assert anomaly_map.ndim == 2
    assert anomaly_map.shape == (20, 30)
    assert np.isfinite(anomaly_map).all()


def test_draem_uses_perlin_mask_and_discriminative_branch():
    import inspect

    import torch

    from pyimgano.models.draem import (
        DRAEMNetwork,
        ImagePathDataset,
        VisionDRAEM,
        _focal_loss,
        _ssim_loss,
    )

    dataset = ImagePathDataset([], anomaly_source_images=[np.zeros((8, 8, 3), dtype=np.uint8)])
    dataset.rng = np.random.default_rng(2)
    original = torch.full((3, 32, 32), 0.2)
    texture = torch.full((3, 32, 32), 0.9)
    augmented, mask = dataset._add_synthetic_anomaly(original, texture)

    assert mask.shape == (1, 32, 32)
    assert torch.any(mask > 0)
    assert not torch.equal(augmented, original)
    assert len(dataset.last_augmentation_indices_) == 3
    assert len(set(dataset.last_augmentation_indices_)) == 3

    pool_input = torch.linspace(0.0, 1.0, 3 * 16 * 16).reshape(3, 16, 16)
    for augmentation_index in range(10):
        pool_output = dataset._apply_texture_augmentation(pool_input, augmentation_index)
        assert pool_output.shape == pool_input.shape
        assert torch.isfinite(pool_output).all()
        assert 0.0 <= float(pool_output.min()) <= float(pool_output.max()) <= 1.0

    network = DRAEMNetwork(base_channels=4)
    assert len(network.reconstructor.encoder_blocks) == 5
    assert len(network.reconstructor.decoder_blocks) == 4
    assert len(network.segmentor.encoder_blocks) == 6
    assert len(network.segmentor.decoder_blocks) == 5
    assert inspect.signature(DRAEMNetwork).parameters["reconstructive_base_channels"].default == 128
    assert inspect.signature(DRAEMNetwork).parameters["discriminative_base_channels"].default == 64
    detector_signature = inspect.signature(VisionDRAEM.__init__)
    assert detector_signature.parameters["epochs"].default == 700
    assert detector_signature.parameters["batch_size"].default == 8
    assert detector_signature.parameters["lr"].default == 0.0001
    reconstruction, logits = network(augmented.unsqueeze(0).repeat(2, 1, 1, 1))
    assert reconstruction.shape == (2, 3, 32, 32)
    assert logits.shape == (2, 2, 32, 32)
    masks = mask.unsqueeze(0).repeat(2, 1, 1, 1)
    loss = (
        torch.nn.functional.mse_loss(reconstruction, original.unsqueeze(0).repeat(2, 1, 1, 1))
        + _ssim_loss(reconstruction, original.unsqueeze(0).repeat(2, 1, 1, 1))
        + _focal_loss(logits, masks)
    )
    loss.backward()

    assert any(parameter.grad is not None for parameter in network.segmentor.parameters())
