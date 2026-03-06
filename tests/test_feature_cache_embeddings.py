from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def _write_rgb(path: Path, *, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.ones((40, 40, 3), dtype=np.uint8) * int(value)
    Image.fromarray(img, mode="RGB").save(path)


def _build_torchscript_checkpoint(path: Path) -> Path:
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
    scripted.save(str(path))
    return path


def test_torchvision_backbone_cache_is_path_only(tmp_path: Path) -> None:
    import pyimgano.features  # noqa: F401
    from pyimgano.features.registry import create_feature_extractor

    img0 = tmp_path / "a.png"
    img1 = tmp_path / "b.png"
    _write_rgb(img0, value=32)
    _write_rgb(img1, value=96)

    cache_dir = tmp_path / "cache_tv"
    ext = create_feature_extractor(
        "torchvision_backbone",
        backbone="resnet18",
        pretrained=False,
        device="cpu",
        pool="avg",
        cache_dir=str(cache_dir),
        batch_size=2,
        image_size=64,
    )

    feats_paths_1 = np.asarray(ext.extract([str(img0), str(img1)]), dtype=np.float64)
    assert feats_paths_1.shape[0] == 2
    assert ext.last_cache_stats_["enabled"] is True
    assert int(ext.last_cache_stats_["hits"]) == 0
    assert int(ext.last_cache_stats_["misses"]) == 2

    feats_paths_2 = np.asarray(ext.extract([str(img0), str(img1)]), dtype=np.float64)
    assert np.allclose(feats_paths_1, feats_paths_2)
    assert ext.last_cache_stats_["enabled"] is True
    assert int(ext.last_cache_stats_["hits"]) == 2
    assert int(ext.last_cache_stats_["misses"]) == 0

    arrays = [
        np.ones((40, 40, 3), dtype=np.uint8) * 32,
        np.ones((40, 40, 3), dtype=np.uint8) * 96,
    ]
    feats_arrays = np.asarray(ext.extract(arrays), dtype=np.float64)
    assert feats_arrays.shape == feats_paths_1.shape
    assert np.isfinite(feats_arrays).all()
    assert ext.last_cache_stats_["enabled"] is False
    assert int(ext.last_cache_stats_["hits"]) == 0
    assert int(ext.last_cache_stats_["misses"]) == 2


def test_torchscript_embed_cache_is_path_only(tmp_path: Path) -> None:
    from pyimgano.features import create_feature_extractor

    ckpt = _build_torchscript_checkpoint(tmp_path / "toy_embed.pt")

    img0 = tmp_path / "c.png"
    img1 = tmp_path / "d.png"
    _write_rgb(img0, value=12)
    _write_rgb(img1, value=180)

    cache_dir = tmp_path / "cache_ts"
    ext = create_feature_extractor(
        "torchscript_embed",
        checkpoint_path=str(ckpt),
        device="cpu",
        image_size=32,
        batch_size=2,
        cache_dir=str(cache_dir),
    )

    feats_paths_1 = np.asarray(ext.extract([str(img0), str(img1)]), dtype=np.float64)
    assert feats_paths_1.shape == (2, 4)
    assert ext.last_cache_stats_["enabled"] is True
    assert int(ext.last_cache_stats_["hits"]) == 0
    assert int(ext.last_cache_stats_["misses"]) == 2

    feats_paths_2 = np.asarray(ext.extract([str(img0), str(img1)]), dtype=np.float64)
    assert np.allclose(feats_paths_1, feats_paths_2)
    assert ext.last_cache_stats_["enabled"] is True
    assert int(ext.last_cache_stats_["hits"]) == 2
    assert int(ext.last_cache_stats_["misses"]) == 0

    arrays = [
        np.ones((40, 40, 3), dtype=np.uint8) * 12,
        np.ones((40, 40, 3), dtype=np.uint8) * 180,
    ]
    feats_arrays = np.asarray(ext.extract(arrays), dtype=np.float64)
    assert feats_arrays.shape == (2, 4)
    assert np.isfinite(feats_arrays).all()
    assert ext.last_cache_stats_["enabled"] is False
    assert int(ext.last_cache_stats_["hits"]) == 0
    assert int(ext.last_cache_stats_["misses"]) == 0
