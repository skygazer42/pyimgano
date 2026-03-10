from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pyimgano.utils.path_normalize import normalize_path

SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _iter_images(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    out: list[Path] = []
    for p in sorted(directory.rglob("*")):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
            out.append(p)
    return out


def _relpath(path: Path, *, base: Path, absolute: bool) -> str:
    if absolute:
        return normalize_path(path.resolve())
    try:
        rel = os.path.relpath(str(path.resolve()), start=str(base.resolve()))
        return normalize_path(rel)
    except Exception:
        return normalize_path(str(path))


def convert_btad_to_manifest(
    *,
    root: str | Path,
    category: str,
    out_path: str | Path,
    absolute_paths: bool = False,
    include_masks: bool = False,  # API compat (ignored; BTAD does not ship masks)
) -> list[dict[str, Any]]:
    """Convert an on-disk BTAD category into a JSONL manifest (paths-first).

    Supported layout:

        root/
          <category>/
            train/ok/*.png
            test/ok/*.png
            test/ko/*.png
    """

    _ = include_masks  # unused (kept for manifest_cli / convert_dataset_to_manifest API symmetry)

    root_path = Path(root)
    cat = str(category).strip()
    if not cat:
        raise ValueError("category must be non-empty for btad conversion")

    category_dir = root_path / cat
    if not category_dir.exists():
        raise FileNotFoundError(f"BTAD category directory not found: {category_dir}")

    out_file = Path(out_path)
    out_dir = out_file.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ok_dir = category_dir / "train" / "ok"
    test_ok_dir = category_dir / "test" / "ok"
    test_ko_dir = category_dir / "test" / "ko"

    records: list[dict[str, Any]] = []

    for p in _iter_images(train_ok_dir):
        records.append(
            {
                "image_path": _relpath(p, base=out_dir, absolute=absolute_paths),
                "category": cat,
                "split": "train",
            }
        )

    for p in _iter_images(test_ok_dir):
        records.append(
            {
                "image_path": _relpath(p, base=out_dir, absolute=absolute_paths),
                "category": cat,
                "split": "test",
                "label": 0,
            }
        )

    for p in _iter_images(test_ko_dir):
        records.append(
            {
                "image_path": _relpath(p, base=out_dir, absolute=absolute_paths),
                "category": cat,
                "split": "test",
                "label": 1,
            }
        )

    # Stable order for reproducible diffs.
    records.sort(key=lambda r: (str(r.get("split", "")), str(r.get("image_path", ""))))

    with out_file.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return records


__all__ = ["convert_btad_to_manifest"]

