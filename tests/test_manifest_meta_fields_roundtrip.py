from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


def _write_rgb(path: Path, *, color: tuple[int, int, int] = (10, 20, 30)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color=color).save(path)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def test_manifest_meta_view_id_and_condition_roundtrip(tmp_path: Path) -> None:
    from pyimgano.datasets.manifest import load_manifest_benchmark_split

    root = tmp_path / "root"
    _write_rgb(root / "train" / "normal" / "n0.png", color=(10, 20, 30))
    _write_rgb(root / "test" / "normal" / "x0.png", color=(11, 21, 31))
    _write_rgb(root / "test" / "anomaly" / "a0.png", color=(200, 10, 10))

    manifest = root / "manifest.jsonl"
    _write_jsonl(
        manifest,
        [
            {
                "image_path": "train/normal/n0.png",
                "category": "demo",
                "split": "train",
            },
            {
                "image_path": "test/normal/x0.png",
                "category": "demo",
                "split": "test",
                "label": 0,
                "meta": {"view_id": "cam0", "condition": "day"},
            },
            {
                "image_path": "test/anomaly/a0.png",
                "category": "demo",
                "split": "test",
                "label": 1,
                "meta": {"view_id": "cam1", "condition": "night"},
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

    assert split.test_meta is not None
    assert len(split.test_meta) == len(split.test_paths)

    # Ensure meta is preserved for test records.
    metas = [m for m in split.test_meta if m is not None]
    assert metas, "expected some meta objects"
    for m in metas:
        assert "view_id" in m
        assert "condition" in m

    # Sanity: image-level split still works.
    assert split.test_labels.shape == (len(split.test_paths),)
    assert np.isin(split.test_labels, [0, 1]).all()

