from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _write_png(path: Path, arr_u8: np.ndarray) -> None:
    from PIL import Image

    img = Image.fromarray(np.asarray(arr_u8, dtype=np.uint8), mode="RGB")
    img.save(str(path))


def test_torchscript_embed_extractor_smoke_and_cache(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    nn = torch.nn

    class ToyEmbed(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))

        def forward(self, x):  # noqa: ANN001, ANN201 - torchscript signature
            y = self.pool(self.conv(x))
            return y.flatten(1)

    model = ToyEmbed().eval()
    example = torch.zeros((1, 3, 32, 32), dtype=torch.float32)
    scripted = torch.jit.trace(model, example)

    ckpt = tmp_path / "toy_embed.pt"
    scripted.save(str(ckpt))

    img0 = (np.random.default_rng(0).integers(0, 255, size=(40, 40, 3))).astype(np.uint8)
    img1 = (np.random.default_rng(1).integers(0, 255, size=(40, 40, 3))).astype(np.uint8)
    p0 = tmp_path / "a.png"
    p1 = tmp_path / "b.png"
    _write_png(p0, img0)
    _write_png(p1, img1)

    from pyimgano.features import create_feature_extractor

    cache_dir = tmp_path / "cache"
    ext = create_feature_extractor(
        "torchscript_embed",
        checkpoint_path=str(ckpt),
        device="cpu",
        image_size=32,
        batch_size=2,
        cache_dir=str(cache_dir),
    )

    feats1 = np.asarray(ext.extract([str(p0), str(p1)]))
    assert feats1.shape == (2, 4)
    assert feats1.dtype == np.float64
    assert np.isfinite(feats1).all()
    assert ext.last_cache_stats_["enabled"] is True
    assert int(ext.last_cache_stats_["hits"]) == 0
    assert int(ext.last_cache_stats_["misses"]) == 2

    feats2 = np.asarray(ext.extract([str(p0), str(p1)]))
    assert feats2.shape == (2, 4)
    assert np.allclose(feats1, feats2)
    assert ext.last_cache_stats_["enabled"] is True
    assert int(ext.last_cache_stats_["hits"]) == 2
    assert int(ext.last_cache_stats_["misses"]) == 0
