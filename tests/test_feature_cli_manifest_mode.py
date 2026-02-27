from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


def _write_rgb(path: Path, *, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color=color).save(path)


def test_feature_cli_manifest_mode_reads_jsonl_and_preserves_order(tmp_path: Path) -> None:
    from pyimgano.feature_cli import main

    root = tmp_path / "imgs"
    a = root / "a.png"
    b = root / "b.png"
    _write_rgb(a, color=(10, 20, 30))
    _write_rgb(b, color=(40, 50, 60))

    manifest = tmp_path / "manifest.jsonl"
    lines = [
        {"image_path": str(a), "category": "c1", "split": "train", "label": 0},
        {"image_path": str(b), "category": "c1", "split": "train", "label": 0},
    ]
    manifest.write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")

    out_npy = tmp_path / "feats.npy"
    out_paths = tmp_path / "paths.json"

    code = main(
        [
            "--manifest",
            str(manifest),
            "--manifest-category",
            "c1",
            "--manifest-split",
            "train",
            "--output",
            str(out_npy),
            "--paths-json",
            str(out_paths),
            "--extractor",
            "color_hist",
            "--extractor-kwargs",
            json.dumps({"colorspace": "rgb", "bins": [2, 2, 2]}),
        ]
    )
    assert code == 0

    feats = np.load(str(out_npy), allow_pickle=False)
    # color_hist concatenates per-channel histograms (sum of bins)
    assert feats.shape == (2, 6)
    assert np.all(np.isfinite(feats))

    paths = json.loads(out_paths.read_text(encoding="utf-8"))
    assert paths == [str(a), str(b)]
