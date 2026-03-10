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


def _resolve_dir(parent: Path, preferred: str, fallbacks: list[str]) -> Path:
    preferred_dir = parent / preferred
    if preferred_dir.exists():
        return preferred_dir
    for name in fallbacks:
        candidate = parent / name
        if candidate.exists():
            return candidate
    return preferred_dir


def _find_mask_for_bad_image(*, bad_image: Path, gt_bad_dir: Path) -> Path | None:
    # Mirror the best-effort candidates used by the benchmark loader (VisADataset).
    candidates: list[Path] = [
        gt_bad_dir / bad_image.name,
        gt_bad_dir / f"{bad_image.stem}.png",
        gt_bad_dir / f"{bad_image.stem}_mask.png",
    ]
    for c in candidates:
        if c.exists() and c.is_file():
            return c
    return None


def convert_visa_to_manifest(
    *,
    root: str | Path,
    category: str,
    out_path: str | Path,
    absolute_paths: bool = False,
    include_masks: bool = False,
) -> list[dict[str, Any]]:
    """Convert an on-disk VisA category into a JSONL manifest (paths-first).

    Supported layouts (common exports):

        root/
          visa_pytorch/
            <category>/
              train/good/*.png
              test/good/*.png
              test/bad/*.png
              ground_truth/bad/*.png   (optional)

    Users may also pass `root` directly as `visa_pytorch/`.
    Folder name fallbacks:
    - good: good|ok|normal
    - bad: bad|ko|anomaly
    """

    root_path = Path(root)
    cat = str(category).strip()
    if not cat:
        raise ValueError("category must be non-empty for visa conversion")

    base_root = root_path / "visa_pytorch" if (root_path / "visa_pytorch").exists() else root_path
    category_dir = base_root / cat
    if not category_dir.exists():
        raise FileNotFoundError(f"VisA category directory not found: {category_dir}")

    out_file = Path(out_path)
    out_dir = out_file.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    train_dir = category_dir / "train"
    train_good_dir = _resolve_dir(train_dir, "good", ["ok", "normal"])

    test_dir = category_dir / "test"
    test_good_dir = _resolve_dir(test_dir, "good", ["ok", "normal"])
    test_bad_dir = _resolve_dir(test_dir, "bad", ["ko", "anomaly"])

    gt_dir = category_dir / "ground_truth"
    gt_bad_dir = _resolve_dir(gt_dir, "bad", ["ko", "anomaly"])

    records: list[dict[str, Any]] = []

    for p in _iter_images(train_good_dir):
        records.append(
            {
                "image_path": _relpath(p, base=out_dir, absolute=absolute_paths),
                "category": cat,
                "split": "train",
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


__all__ = ["convert_visa_to_manifest"]

