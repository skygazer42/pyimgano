from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def _write_rgb(path: Path, *, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.ones((32, 32, 3), dtype=np.uint8) * int(value)
    Image.fromarray(img, mode="RGB").save(path)


@pytest.mark.parametrize("pool", ["avg", "max", "gem"])
def test_torchvision_backbone_pooling_modes_resnet18(tmp_path: Path, pool: str) -> None:
    import pyimgano.features  # noqa: F401 - side effects (registry)
    from pyimgano.features.registry import create_feature_extractor

    root = tmp_path / "imgs"
    _write_rgb(root / "a.png", value=10)
    _write_rgb(root / "b.png", value=20)

    extractor = create_feature_extractor(
        "torchvision_backbone",
        backbone="resnet18",
        pretrained=False,
        device="cpu",
        pool=str(pool),
        batch_size=2,
        image_size=64,
    )
    feats = np.asarray(extractor.extract([str(root / "a.png"), str(root / "b.png")]))
    assert feats.shape[0] == 2
    assert feats.shape[1] > 1
    assert np.isfinite(feats).all()


def test_torchvision_backbone_pool_cls_vit(tmp_path: Path) -> None:
    import pyimgano.features  # noqa: F401
    from pyimgano.features.registry import create_feature_extractor

    root = tmp_path / "imgs"
    _write_rgb(root / "a.png", value=10)
    _write_rgb(root / "b.png", value=20)

    extractor = create_feature_extractor(
        "torchvision_backbone",
        backbone="vit_b_16",
        pretrained=False,
        device="cpu",
        pool="cls",
        batch_size=1,
    )
    feats = np.asarray(extractor.extract([str(root / "a.png"), str(root / "b.png")]))
    assert feats.shape[0] == 2
    assert feats.shape[1] > 1
    assert np.isfinite(feats).all()


def test_torchvision_backbone_rejects_unknown_pool_mode() -> None:
    import pyimgano.features  # noqa: F401
    from pyimgano.features.registry import create_feature_extractor

    with pytest.raises(ValueError, match="pool"):
        _ = create_feature_extractor(
            "torchvision_backbone",
            backbone="resnet18",
            pretrained=False,
            device="cpu",
            pool="nope",
        )
