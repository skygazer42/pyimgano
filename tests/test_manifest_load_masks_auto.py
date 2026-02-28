from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


def _write_rgb(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color=(10, 20, 30)).save(path)


def _write_mask(path: Path, *, value: int = 255) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((16, 16), int(value), dtype=np.uint8), mode="L").save(path)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def test_load_masks_auto_skips_when_no_mask_paths(tmp_path: Path) -> None:
    from pyimgano.datasets.manifest import load_manifest_benchmark_split

    root = tmp_path / "root"
    _write_rgb(root / "train.png")
    _write_rgb(root / "test.png")

    manifest = root / "manifest.jsonl"
    _write_jsonl(
        manifest,
        [
            {"image_path": "train.png", "category": "demo", "split": "train"},
            {"image_path": "test.png", "category": "demo", "split": "test", "label": 0},
        ],
    )

    split = load_manifest_benchmark_split(
        manifest_path=manifest,
        root_fallback=root,
        category="demo",
        resize=(8, 8),
        load_masks="auto",
    )

    assert split.test_masks is None
    assert split.pixel_skip_reason is not None
    assert "load_masks=auto" in str(split.pixel_skip_reason)


def test_load_masks_auto_loads_when_masks_present(tmp_path: Path) -> None:
    from pyimgano.datasets.manifest import load_manifest_benchmark_split

    root = tmp_path / "root"
    _write_rgb(root / "train.png")
    _write_rgb(root / "a.png")
    _write_mask(root / "a_mask.png", value=255)

    manifest = root / "manifest.jsonl"
    _write_jsonl(
        manifest,
        [
            {"image_path": "train.png", "category": "demo", "split": "train"},
            {
                "image_path": "a.png",
                "category": "demo",
                "split": "test",
                "label": 1,
                "mask_path": "a_mask.png",
            },
        ],
    )

    split = load_manifest_benchmark_split(
        manifest_path=manifest,
        root_fallback=root,
        category="demo",
        resize=(8, 8),
        load_masks="auto",
    )

    assert split.test_masks is not None
    assert split.test_masks.shape == (len(split.test_paths), 8, 8)

