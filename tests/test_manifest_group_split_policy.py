from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


def _write_rgb(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=(10, 20, 30)).save(path)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def test_manifest_group_aware_split_uses_meta_group_id(tmp_path: Path) -> None:
    from pyimgano.datasets.manifest import load_manifest_benchmark_split

    root = tmp_path / "root"
    _write_rgb(root / "n0.png")
    _write_rgb(root / "n1.png")
    _write_rgb(root / "a0.png")

    manifest = root / "manifest.jsonl"
    _write_jsonl(
        manifest,
        [
            {
                "image_path": "n0.png",
                "category": "demo",
                "split": "train",
                "meta": {"group_id": "g1"},
            },
            {
                "image_path": "n1.png",
                "category": "demo",
                # no explicit split; should follow the group
                "meta": {"group_id": "g1"},
            },
            {
                "image_path": "a0.png",
                "category": "demo",
                "split": "test",
                "label": 1,
                "meta": {"group_id": "g2"},
            },
        ],
    )

    split = load_manifest_benchmark_split(
        manifest_path=manifest,
        root_fallback=root,
        category="demo",
        resize=(8, 8),
        load_masks=False,
    )

    train_names = {Path(p).name for p in split.train_paths}
    test_names = {Path(p).name for p in split.test_paths}

    assert {"n0.png", "n1.png"} <= train_names, "records with same meta.group_id must stay together"
    assert "a0.png" in test_names

