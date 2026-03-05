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


def test_manifest_validate_ok(tmp_path: Path) -> None:
    from pyimgano.datasets.manifest_validate import validate_manifest_file

    root = tmp_path / "root"
    _write_rgb(root / "n0.png")
    manifest = root / "manifest.jsonl"
    _write_jsonl(
        manifest,
        [{"image_path": "n0.png", "category": "demo", "split": "train"}],
    )

    report = validate_manifest_file(
        manifest_path=manifest,
        root_fallback=root,
        check_files=True,
        category="demo",
    )
    assert report.ok is True
    assert report.errors == []


def test_manifest_validate_missing_file_is_error(tmp_path: Path) -> None:
    from pyimgano.datasets.manifest_validate import validate_manifest_file

    root = tmp_path / "root"
    manifest = root / "manifest.jsonl"
    _write_jsonl(
        manifest,
        [{"image_path": "missing.png", "category": "demo", "split": "train"}],
    )

    report = validate_manifest_file(
        manifest_path=manifest,
        root_fallback=root,
        check_files=True,
        category="demo",
    )
    assert report.ok is False
    assert report.errors
