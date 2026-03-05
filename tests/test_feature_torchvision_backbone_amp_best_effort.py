from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def _write_rgb(path: Path, *, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.ones((32, 32, 3), dtype=np.uint8) * int(value)
    Image.fromarray(img, mode="RGB").save(path)


def test_torchvision_backbone_amp_best_effort_on_cpu(tmp_path: Path) -> None:
    """amp=True should be safe on CPU (best-effort no-op)."""

    import pyimgano.features  # noqa: F401
    from pyimgano.features.registry import create_feature_extractor

    root = tmp_path / "imgs"
    _write_rgb(root / "a.png", value=10)
    _write_rgb(root / "b.png", value=20)

    extractor = create_feature_extractor(
        "torchvision_backbone",
        backbone="resnet18",
        pretrained=False,
        device="cpu",
        amp=True,
        batch_size=2,
        image_size=64,
    )
    feats = np.asarray(extractor.extract([str(root / "a.png"), str(root / "b.png")]))
    assert feats.shape[0] == 2
    assert feats.shape[1] > 1
    assert np.isfinite(feats).all()
