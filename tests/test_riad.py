from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")


def _make_rgb_batch(*, count: int = 4, size: int = 32) -> list[np.ndarray]:
    rng = np.random.default_rng(3)
    out: list[np.ndarray] = []
    for _ in range(count):
        out.append(rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8))
    return out


def test_riad_contract_accepts_numpy_image_list() -> None:
    from pyimgano.models import create_model

    train = _make_rgb_batch(count=2, size=16)
    test = _make_rgb_batch(count=1, size=16)

    det = create_model(
        "vision_riad",
        image_size=(16, 16),
        region_sizes=(16,),
        num_disjoint_masks=1,
        epochs=1,
        batch_size=2,
        gaussian_sigma=0,
        device="cpu",
        random_state=0,
    )

    det.fit(train)
    scores = np.asarray(det.decision_function(test), dtype=np.float64).reshape(-1)
    assert scores.shape == (1,)
    assert np.all(np.isfinite(scores))


def test_riad_fit_does_not_print_progress(capsys) -> None:
    from pyimgano.models import create_model

    train = _make_rgb_batch(count=2, size=16)

    det = create_model(
        "vision_riad",
        image_size=(16, 16),
        region_sizes=(16,),
        num_disjoint_masks=1,
        epochs=1,
        batch_size=2,
        device="cpu",
        random_state=0,
    )

    det.fit(train)
    out = capsys.readouterr().out
    assert out == ""


def test_riad_disjoint_masks_partition_every_pixel_once() -> None:
    from pyimgano.models.riad import ImageDecomposer

    decomposer = ImageDecomposer(num_disjoint_masks=3, random_state=5)
    masks = decomposer.create_disjoint_masks((31, 29), region_size=8)

    assert masks.shape == (3, 1, 31, 29)
    assert set(np.unique(masks)) == {0.0, 1.0}
    assert np.array_equal((1.0 - masks).sum(axis=0), np.ones((1, 31, 29)))


def test_riad_network_matches_paper_channel_pyramid() -> None:
    import torch

    from pyimgano.models.riad import UNet

    model = UNet()
    assert [block.conv1.out_channels for block in model.down_blocks] == [64, 128, 256, 512, 512]
    assert [block.conv1.stride for block in model.down_blocks] == [
        (1, 1),
        (2, 2),
        (2, 2),
        (2, 2),
        (2, 2),
    ]
    assert [block.up.out_channels for block in model.up_blocks] == [512, 256, 128, 64]

    model.eval()
    output = model(torch.zeros(1, 3, 16, 16))
    assert output.shape == (1, 3, 16, 16)
    assert torch.all(output >= -1.0) and torch.all(output <= 1.0)


def test_riad_msgms_identity_and_change() -> None:
    import torch

    from pyimgano.models.riad import MSGMSLoss

    metric = MSGMSLoss(num_scales=4)
    image = torch.zeros(1, 3, 32, 32)
    changed = image.clone()
    changed[:, :, 8:24, 8:24] = 1.0

    identical = metric(image, image, as_loss=False)
    different = metric(image, changed, as_loss=False)
    assert identical.shape == (1, 1, 32, 32)
    assert torch.allclose(identical, torch.zeros_like(identical), atol=1e-6)
    assert different.max() > 0
