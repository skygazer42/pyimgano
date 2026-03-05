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


def _find_mask_for_bad_image(*, bad_image: Path, gt_bad_dir: Path) -> Path | None:
    # Mirror the best-effort candidates used by the paths-first loader.
    candidates: list[Path] = [
        gt_bad_dir / bad_image.name,
        gt_bad_dir / f"{bad_image.stem}_mask{bad_image.suffix}",
        gt_bad_dir / f"{bad_image.stem}_mask.png",
        gt_bad_dir / f"{bad_image.stem}.png",
    ]
    for c in candidates:
        if c.exists() and c.is_file():
            return c
    return None


def convert_mvtec_ad2_to_manifest(
    *,
    root: str | Path,
    category: str,
    out_path: str | Path,
    split: str = "test_public",
    absolute_paths: bool = False,
    include_masks: bool = False,
) -> list[dict[str, Any]]:
    """Convert an on-disk MVTec AD 2 category into a JSONL manifest (paths-first)."""

    root_path = Path(root)
    cat = str(category).strip()
    if not cat:
        raise ValueError("category must be non-empty for mvtec_ad2 conversion")

    category_dir = root_path / cat
    if not category_dir.exists():
        raise FileNotFoundError(f"MVTec AD 2 category directory not found: {category_dir}")

    out_file = Path(out_path)
    out_dir = out_file.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    train_good_dir = category_dir / "train" / "good"
    val_good_dir = category_dir / "validation" / "good"

    split_dir = category_dir / str(split)
    test_good_dir = split_dir / "good"
    test_bad_dir = split_dir / "bad"
    gt_bad_dir = split_dir / "ground_truth" / "bad"

    records: list[dict[str, Any]] = []

    for p in _iter_images(train_good_dir):
        records.append(
            {
                "image_path": _relpath(p, base=out_dir, absolute=absolute_paths),
                "category": cat,
                "split": "train",
            }
        )

    for p in _iter_images(val_good_dir):
        records.append(
            {
                "image_path": _relpath(p, base=out_dir, absolute=absolute_paths),
                "category": cat,
                "split": "val",
            }
        )

    for p in _iter_images(test_good_dir):
        records.append(
            {
                "image_path": _relpath(p, base=out_dir, absolute=absolute_paths),
                "category": cat,
                "split": "test",
                "label": 0,
            }
        )

    for p in _iter_images(test_bad_dir):
        rec: dict[str, Any] = {
            "image_path": _relpath(p, base=out_dir, absolute=absolute_paths),
            "category": cat,
            "split": "test",
            "label": 1,
        }
        if include_masks and gt_bad_dir.exists():
            mask = _find_mask_for_bad_image(bad_image=p, gt_bad_dir=gt_bad_dir)
            if mask is not None:
                rec["mask_path"] = _relpath(mask, base=out_dir, absolute=absolute_paths)
        records.append(rec)

    # Stable order for reproducible diffs.
    records.sort(key=lambda r: (str(r.get("split", "")), str(r.get("image_path", ""))))

    with out_file.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return records


__all__ = ["convert_mvtec_ad2_to_manifest"]
