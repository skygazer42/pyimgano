from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def test_realnet_paper_defaults_and_image_score() -> None:
    from pyimgano.models.realnet import VisionRealNet

    detector = VisionRealNet(pretrained=False, device="cpu")

    assert detector.backbone == "wide_resnet50_2"
    assert detector.selected_channels == (256, 512, 512, 256)
    assert detector.rrs_modes == ("max", "mean")
    assert detector.rrs_mode_numbers == (256, 256)
    assert detector.batch_size == 16
    assert detector.epochs == 1000
    assert detector.afs_batches == 64
    assert detector.image_size == 256
    assert detector.learning_rate == pytest.approx(1e-4)

    anomaly_map = torch.zeros((1, 1, 20, 20))
    anomaly_map[:, :, :16, :16] = 0.5
    np.testing.assert_allclose(detector._image_scores(anomaly_map), [0.5])


def test_realnet_refuses_to_fake_missing_sdas_pairs() -> None:
    from pyimgano.models.realnet import VisionRealNet

    detector = VisionRealNet(pretrained=False, device="cpu", epochs=1, afs_batches=1)
    normal = np.zeros((2, 16, 16, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="online SDAS"):
        detector.fit(normal)
