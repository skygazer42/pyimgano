from __future__ import annotations

from pathlib import Path

import numpy as np


def _write_png(path: Path, *, value: int) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.ones((16, 16, 3), dtype=np.uint8) * int(value)
    Image.fromarray(img, mode="RGB").save(path)


def test_feature_cli_cache_dir_reuses_cache(tmp_path: Path, monkeypatch) -> None:
    from pyimgano.feature_cli import main as feature_main

    root = tmp_path / "imgs"
    _write_png(root / "a.png", value=50)
    _write_png(root / "b.png", value=51)

    cache_dir = tmp_path / "cache"
    out1 = tmp_path / "feats1.npy"
    out2 = tmp_path / "feats2.npy"

    # First pass: fills the cache.
    rc1 = feature_main(
        [
            "--root",
            str(root),
            "--pattern",
            "*.png",
            "--extractor",
            "edge_stats",
            "--extractor-kwargs",
            "{}",
            "--cache-dir",
            str(cache_dir),
            "--output",
            str(out1),
        ]
    )
    assert rc1 == 0
    assert out1.exists()

    # Second pass: should be able to complete without calling the underlying extractor.
    from pyimgano.features.edge_stats import EdgeStatsExtractor

    def _boom(self, inputs):  # noqa: ANN001, ANN201 - signature matches extractor protocol
        del inputs, self
        raise RuntimeError("EdgeStatsExtractor.extract should not be called when cache is warm")

    monkeypatch.setattr(EdgeStatsExtractor, "extract", _boom, raising=True)

    rc2 = feature_main(
        [
            "--root",
            str(root),
            "--pattern",
            "*.png",
            "--extractor",
            "edge_stats",
            "--extractor-kwargs",
            "{}",
            "--cache-dir",
            str(cache_dir),
            "--output",
            str(out2),
        ]
    )
    assert rc2 == 0
    assert out2.exists()

    f1 = np.load(str(out1), allow_pickle=False)
    f2 = np.load(str(out2), allow_pickle=False)
    assert f1.shape == f2.shape
    assert np.allclose(f1, f2)
