from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")


def test_torchvision_vit_tokens_extractor_cls_pool_shape() -> None:
    from pyimgano.features import create_feature_extractor

    ext = create_feature_extractor(
        "torchvision_vit_tokens",
        backbone="vit_b_16",
        pretrained=False,
        pool="cls",
        device="cpu",
        batch_size=2,
        image_size=224,
    )

    imgs = [
        np.zeros((64, 64, 3), dtype=np.uint8),
        np.full((64, 64, 3), 255, dtype=np.uint8),
    ]
    feats = np.asarray(ext.extract(imgs))

    assert feats.ndim == 2
    assert feats.shape[0] == 2
    # vit_b_16 hidden dim is 768 in torchvision.
    assert feats.shape[1] == 768
    assert np.all(np.isfinite(feats))


def test_torchvision_vit_tokens_extractor_mean_pool_shape() -> None:
    from pyimgano.features import create_feature_extractor

    ext = create_feature_extractor(
        "torchvision_vit_tokens",
        backbone="vit_b_16",
        pretrained=False,
        pool="mean",
        device="cpu",
        batch_size=2,
        image_size=224,
    )

    imgs = [
        np.zeros((64, 64, 3), dtype=np.uint8),
        np.full((64, 64, 3), 127, dtype=np.uint8),
    ]
    feats = np.asarray(ext.extract(imgs))

    assert feats.ndim == 2
    assert feats.shape[0] == 2
    assert feats.shape[1] == 768
    assert np.all(np.isfinite(feats))
