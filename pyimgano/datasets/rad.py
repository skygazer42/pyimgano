from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

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


def _extract_rad_meta_best_effort(path: Path) -> Mapping[str, Any] | None:
    """Extract (view_id, condition) from path components when present.

    RAD-like datasets are multi-view; we avoid guessing too much and only emit
    metadata when directory names clearly encode it.
    """

    view_id = None
    condition = None
    parts = [p.lower() for p in path.parts]
    for part in parts:
        if view_id is None and (part.startswith("view_") or part.startswith("view-")):
            view_id = part.split("_", 1)[-1].split("-", 1)[-1]
        if view_id is None and (part.startswith("cam_") or part.startswith("cam-")):
            view_id = part.split("_", 1)[-1].split("-", 1)[-1]
        if condition is None and (part.startswith("cond_") or part.startswith("cond-")):
            condition = part.split("_", 1)[-1].split("-", 1)[-1]
        if condition is None and (part.startswith("light_") or part.startswith("light-")):
            condition = part.split("_", 1)[-1].split("-", 1)[-1]

    meta: dict[str, Any] = {}
    if view_id is not None and str(view_id).strip():
        meta["view_id"] = str(view_id)
    if condition is not None and str(condition).strip():
        meta["condition"] = str(condition)
    return meta or None


def convert_rad_to_manifest(
    *,
    root: str | Path,
    out_path: str | Path,
    category: str = "rad",
    absolute_paths: bool = False,
    include_masks: bool = False,
) -> list[dict[str, Any]]:
    """Convert a RAD-like dataset tree into a JSONL manifest (paths-first).

    Supported layouts (best-effort):
    - custom-like: train/normal, test/normal, test/anomaly, ground_truth/anomaly
    - mvtec-like: train/good, test/good, test/bad, ground_truth/bad

    Multi-view metadata is captured into `meta.view_id` / `meta.condition` when
    it is encoded in directory names.
    """

    root_path = Path(root)
    out_file = Path(out_path)
    out_dir = out_file.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    cat = str(category).strip() or "rad"

    # Custom-like.
    train_dir = root_path / "train" / "normal"
    test_normal_dir = root_path / "test" / "normal"
    test_anomaly_dir = root_path / "test" / "anomaly"
    gt_dir = root_path / "ground_truth" / "anomaly"

    # MVTec-like fallback.
    if not (train_dir.exists() and test_normal_dir.exists() and test_anomaly_dir.exists()):
        train_dir = root_path / "train" / "good"
        test_normal_dir = root_path / "test" / "good"
        test_anomaly_dir = root_path / "test" / "bad"
        gt_dir = root_path / "ground_truth" / "bad"

    if not (train_dir.exists() and test_normal_dir.exists() and test_anomaly_dir.exists()):
        raise ValueError(
            "Unable to recognize a supported RAD-style layout.\n"
            "Supported patterns include:\n"
            "- <root>/train/normal + <root>/test/normal + <root>/test/anomaly (+ ground_truth/anomaly)\n"
            "- <root>/train/good + <root>/test/good + <root>/test/bad (+ ground_truth/bad)\n"
            f"root={root_path}"
        )

    records: list[dict[str, Any]] = []
    for p in _iter_images(train_dir):
        rec: dict[str, Any] = {
            "image_path": _relpath(p, base=out_dir, absolute=absolute_paths),
            "category": cat,
            "split": "train",
        }
        meta = _extract_rad_meta_best_effort(p)
        if meta is not None:
            rec["meta"] = dict(meta)
        records.append(rec)

    for p in _iter_images(test_normal_dir):
        rec = {
            "image_path": _relpath(p, base=out_dir, absolute=absolute_paths),
            "category": cat,
            "split": "test",
            "label": 0,
        }
        meta = _extract_rad_meta_best_effort(p)
        if meta is not None:
            rec["meta"] = dict(meta)
        records.append(rec)

    for p in _iter_images(test_anomaly_dir):
        rec = {
            "image_path": _relpath(p, base=out_dir, absolute=absolute_paths),
            "category": cat,
            "split": "test",
            "label": 1,
        }
        meta = _extract_rad_meta_best_effort(p)
        if meta is not None:
            rec["meta"] = dict(meta)

        if include_masks and gt_dir.exists():
            cand = gt_dir / f"{p.stem}_mask.png"
            if not cand.exists():
                cand = gt_dir / f"{p.stem}.png"
            if cand.exists() and cand.is_file():
                rec["mask_path"] = _relpath(cand, base=out_dir, absolute=absolute_paths)

        records.append(rec)

    records.sort(key=lambda r: (str(r.get("split", "")), str(r.get("image_path", ""))))

    with out_file.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return records


__all__ = ["convert_rad_to_manifest"]
