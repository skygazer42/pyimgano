from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image


def _write_rgb(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color=(10, 20, 30)).save(path)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def test_missing_masks_policy_skip(tmp_path: Path) -> None:
    from pyimgano.datasets.manifest import load_manifest_benchmark_split

    root = tmp_path / "root"
    _write_rgb(root / "train.png")
    _write_rgb(root / "a.png")

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
                # missing mask_path on purpose
            },
        ],
    )

    split = load_manifest_benchmark_split(
        manifest_path=manifest,
        root_fallback=root,
        category="demo",
        resize=(8, 8),
        load_masks=True,
        missing_mask_policy="skip",
    )
    assert split.test_masks is None
    assert split.pixel_skip_reason is not None


def test_missing_masks_policy_error(tmp_path: Path) -> None:
    from pyimgano.datasets.manifest import load_manifest_benchmark_split

    root = tmp_path / "root"
    _write_rgb(root / "train.png")
    _write_rgb(root / "a.png")

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
            },
        ],
    )

    with pytest.raises(ValueError, match="Missing mask_path"):
        load_manifest_benchmark_split(
            manifest_path=manifest,
            root_fallback=root,
            category="demo",
            resize=(8, 8),
            load_masks=True,
            missing_mask_policy="error",
        )
