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


def test_manifest_cli_validate_ok(tmp_path: Path) -> None:
    from pyimgano.manifest_cli import main as manifest_main

    root = tmp_path / "root"
    _write_rgb(root / "n0.png")
    manifest = root / "manifest.jsonl"
    _write_jsonl(manifest, [{"image_path": "n0.png", "category": "demo", "split": "train"}])

    rc = manifest_main(
        ["--validate", "--manifest", str(manifest), "--root", str(root), "--category", "demo"]
    )
    assert rc == 0


def test_manifest_cli_validate_reports_error(tmp_path: Path) -> None:
    from pyimgano.manifest_cli import main as manifest_main

    root = tmp_path / "root"
    manifest = root / "manifest.jsonl"
    _write_jsonl(manifest, [{"image_path": "missing.png", "category": "demo", "split": "train"}])

    rc = manifest_main(
        ["--validate", "--manifest", str(manifest), "--root", str(root), "--category", "demo"]
    )
    assert rc == 1
