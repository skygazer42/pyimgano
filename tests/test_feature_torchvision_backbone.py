import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")


def test_torchvision_backbone_extractor_numpy_inputs_smoke() -> None:
    from pyimgano.features.torchvision_backbone import TorchvisionBackboneExtractor

    rng = np.random.RandomState(0)
    imgs = [
        rng.randint(0, 255, size=(64, 64, 3), dtype=np.uint8),
        rng.randint(0, 255, size=(64, 64, 3), dtype=np.uint8),
    ]

    ext = TorchvisionBackboneExtractor(
        backbone="resnet18",
        pretrained=False,
        device="cpu",
        batch_size=2,
        image_size=64,
        input_color="rgb",
    )
    feats = ext.extract(imgs)

    assert feats.shape[0] == 2
    assert feats.ndim == 2
    assert feats.shape[1] >= 32
    assert np.all(np.isfinite(feats))
