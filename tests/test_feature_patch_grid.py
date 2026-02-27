import numpy as np


def test_patch_grid_extractor_numpy_inputs_smoke() -> None:
    from pyimgano.features.patch_grid import PatchGridExtractor

    rng = np.random.RandomState(2)
    imgs = [
        rng.randint(0, 255, size=(64, 64, 3), dtype=np.uint8),
        rng.randint(0, 255, size=(64, 64, 3), dtype=np.uint8),
    ]

    ext = PatchGridExtractor(
        backbone="resnet18",
        node="layer4",
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

