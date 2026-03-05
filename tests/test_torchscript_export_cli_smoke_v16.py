from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def test_torchscript_export_cli_smoke(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")

    out = tmp_path / "resnet18_embed.pt"

    from pyimgano.torchscript_export_cli import main

    rc = main(
        [
            "--backbone",
            "resnet18",
            "--no-pretrained",
            "--image-size",
            "32",
            "--device",
            "cpu",
            "--method",
            "trace",
            "--optimize",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    assert out.exists()
    assert out.stat().st_size > 0

    from pyimgano.features import create_feature_extractor

    ext = create_feature_extractor(
        "torchscript_embed",
        checkpoint_path=str(out),
        device="cpu",
        image_size=32,
        batch_size=2,
    )

    rng = np.random.default_rng(0)
    x0 = rng.integers(0, 255, size=(40, 40, 3), dtype=np.uint8)
    x1 = rng.integers(0, 255, size=(40, 40, 3), dtype=np.uint8)
    feats = np.asarray(ext.extract([x0, x1]))
    assert feats.ndim == 2
    assert feats.shape[0] == 2
    assert feats.shape[1] > 0
    assert feats.dtype == np.float64
    assert np.isfinite(feats).all()
