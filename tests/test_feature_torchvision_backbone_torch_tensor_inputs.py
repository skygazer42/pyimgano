import numpy as np


def test_torchvision_backbone_extractor_torch_tensor_inputs_smoke() -> None:
    import torch

    from pyimgano.features.torchvision_backbone import TorchvisionBackboneExtractor

    rng = np.random.RandomState(0)
    # CHW uint8 (common predecoded tensor format)
    img0 = torch.as_tensor(rng.randint(0, 255, size=(3, 64, 64), dtype=np.uint8))
    # HWC float32 in [0,1]
    img1 = torch.as_tensor(rng.rand(64, 64, 3).astype(np.float32))

    ext = TorchvisionBackboneExtractor(
        backbone="resnet18",
        pretrained=False,
        device="cpu",
        batch_size=2,
        image_size=64,
        input_color="rgb",
    )
    feats = ext.extract([img0, img1])

    assert feats.shape[0] == 2
    assert feats.ndim == 2
    assert feats.shape[1] >= 32
    assert np.all(np.isfinite(feats))

