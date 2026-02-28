from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def _write_rgb(path: Path, *, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.ones((32, 32, 3), dtype=np.uint8) * int(value)
    Image.fromarray(img, mode="RGB").save(path)


def test_torchvision_backbone_cache_dir_creates_cached_embeddings(tmp_path: Path) -> None:
    import pyimgano.features  # noqa: F401
    from pyimgano.features.registry import create_feature_extractor

    root = tmp_path / "imgs"
    a = root / "a.png"
    b = root / "b.png"
    _write_rgb(a, value=10)
    _write_rgb(b, value=20)

    cache_dir = tmp_path / "cache"
    extractor = create_feature_extractor(
        "torchvision_backbone",
        backbone="resnet18",
        pretrained=False,
        device="cpu",
        pool="avg",
        cache_dir=str(cache_dir),
        batch_size=2,
        image_size=64,
    )

    feats1 = np.asarray(extractor.extract([str(a), str(b)]), dtype=np.float64)
    assert feats1.shape[0] == 2
    assert feats1.shape[1] > 1

    cached = list(cache_dir.rglob("*.npy"))
    assert cached, "Expected cached embedding rows under cache_dir"

    # Second call should reuse cache (best-effort). We don't use timing asserts;
    # instead we require some cache hits are reported.
    feats2 = np.asarray(extractor.extract([str(a), str(b)]), dtype=np.float64)
    assert np.allclose(feats2, feats1)

    stats = getattr(extractor, "last_cache_stats_", None)
    assert isinstance(stats, dict)
    assert int(stats.get("hits", 0)) >= 1

