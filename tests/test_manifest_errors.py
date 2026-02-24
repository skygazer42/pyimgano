from __future__ import annotations

import json
from pathlib import Path

import pytest

from pyimgano.datasets.manifest import iter_manifest_records, load_manifest_benchmark_split


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    lines = [json.dumps(row, ensure_ascii=False) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_manifest_error_invalid_split_includes_linenumber(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    _write_jsonl(manifest, [{"image_path": "a.png", "category": "c", "split": "trainn"}])

    with pytest.raises(ValueError, match=r"Manifest line 1: invalid split"):
        list(iter_manifest_records(manifest))


def test_manifest_error_invalid_label_includes_linenumber(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    _write_jsonl(manifest, [{"image_path": "a.png", "category": "c", "label": 2}])

    with pytest.raises(ValueError, match=r"Manifest line 1: label must be 0/1"):
        list(iter_manifest_records(manifest))


def test_manifest_error_split_test_requires_label(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    _write_jsonl(manifest, [{"image_path": "a.png", "category": "c", "split": "test"}])

    with pytest.raises(ValueError, match=r"split='test' requires an explicit label"):
        list(iter_manifest_records(manifest))


def test_manifest_error_conflicting_group_splits(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    _write_jsonl(
        manifest,
        [
            {"image_path": "a.png", "category": "c", "group_id": "g", "split": "train"},
            {"image_path": "b.png", "category": "c", "group_id": "g", "split": "test", "label": 0},
        ],
    )

    with pytest.raises(ValueError, match=r"conflicting explicit splits"):
        load_manifest_benchmark_split(
            manifest_path=manifest,
            root_fallback=None,
            category="c",
            load_masks=False,
        )
