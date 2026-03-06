import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")


def test_torchvision_multilayer_extractor_numpy_inputs_smoke() -> None:
    from pyimgano.features.torchvision_multilayer import TorchvisionMultiLayerExtractor

    rng = np.random.RandomState(1)
    imgs = [
        rng.randint(0, 255, size=(64, 64, 3), dtype=np.uint8),
        rng.randint(0, 255, size=(64, 64, 3), dtype=np.uint8),
    ]

    ext = TorchvisionMultiLayerExtractor(
        backbone="resnet18",
        return_nodes=["layer1", "layer2", "layer3", "layer4"],
        pretrained=False,
        device="cpu",
        batch_size=2,
        image_size=64,
        input_color="rgb",
    )
    feats = ext.extract(imgs)

    assert feats.shape[0] == 2
    assert feats.ndim == 2
    assert feats.shape[1] >= 64
    assert np.all(np.isfinite(feats))
