from __future__ import annotations

import numpy as np


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
    except Exception:
        return False
    return True


def test_torchvision_backbone_gem_extractor_smoke() -> None:
    import pytest

    if not _torch_available():
        pytest.skip("torch/torchvision is not installed")

    from pyimgano.features import create_feature_extractor

    ext = create_feature_extractor(
        "torchvision_backbone_gem",
        backbone="resnet18",
        pretrained=False,
        device="cpu",
        batch_size=2,
        image_size=32,
        node="layer4",
        gem_p=3.0,
        gem_eps=1e-6,
    )

    rng = np.random.default_rng(0)
    imgs = [rng.integers(0, 255, size=(32, 32, 3), dtype=np.uint8) for _ in range(2)]
    feats = np.asarray(ext.extract(imgs), dtype=np.float64)

    assert feats.ndim == 2
    assert feats.shape[0] == 2
    # ResNet18 layer4 has 512 channels => (N,512) embedding.
    assert feats.shape[1] == 512
    assert np.all(np.isfinite(feats))

