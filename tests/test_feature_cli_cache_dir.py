from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


def _write_rgb(path: Path, *, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color=color).save(path)


def test_feature_cli_cache_dir_creates_cached_rows(tmp_path: Path) -> None:
    from pyimgano.feature_cli import main

    root = tmp_path / "imgs"
    _write_rgb(root / "a.png", color=(10, 10, 10))
    _write_rgb(root / "b.png", color=(20, 20, 20))

    cache_dir = tmp_path / "cache"
    out_npy1 = tmp_path / "feats1.npy"

    code1 = main(
        [
            "--root",
            str(root),
            "--pattern",
            "*.png",
            "--output",
            str(out_npy1),
            "--extractor",
            "color_hist",
            "--extractor-kwargs",
            json.dumps({"colorspace": "rgb", "bins": [2, 2, 2]}),
            "--cache-dir",
            str(cache_dir),
        ]
    )
    assert code1 == 0
    feats1 = np.load(str(out_npy1), allow_pickle=False)
    assert feats1.shape == (2, 6)

    cached = list(cache_dir.rglob("*.npy"))
    assert cached, "Expected cached .npy rows under --cache-dir"

    # Second run should still succeed and reuse the cache (best-effort).
    out_npy2 = tmp_path / "feats2.npy"
    code2 = main(
        [
            "--root",
            str(root),
            "--pattern",
            "*.png",
            "--output",
            str(out_npy2),
            "--extractor",
            "color_hist",
            "--extractor-kwargs",
            json.dumps({"colorspace": "rgb", "bins": [2, 2, 2]}),
            "--cache-dir",
            str(cache_dir),
        ]
    )
    assert code2 == 0
    feats2 = np.load(str(out_npy2), allow_pickle=False)
    assert np.allclose(feats2, feats1)

