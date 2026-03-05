from __future__ import annotations

import json
import os
from dataclasses import dataclass
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


def _extract_meta_best_effort(path: Path) -> Mapping[str, Any] | None:
    """Best-effort metadata extraction for multi-view/condition datasets.

    This is intentionally conservative: it only emits fields when clearly present
    in directory names. The goal is to preserve useful grouping hints without
    guessing too much.
    """

    view_id = None
    condition = None
    parts = [p.lower() for p in path.parts]
    for part in parts:
        if view_id is None and (part.startswith("view_") or part.startswith("view-")):
            view_id = part.split("_", 1)[-1].split("-", 1)[-1]
        if view_id is None and part.startswith("cam"):
            view_id = part
        if condition is None and (part.startswith("cond_") or part.startswith("cond-")):
            condition = part.split("_", 1)[-1].split("-", 1)[-1]
        if condition is None and part.startswith("lighting"):
            condition = part

    meta: dict[str, Any] = {}
    if view_id is not None and str(view_id).strip():
        meta["view_id"] = str(view_id)
    if condition is not None and str(condition).strip():
        meta["condition"] = str(condition)
    return meta or None


@dataclass(frozen=True)
class RealIADLayout:
    train_normal: Path
    test_normal: Path
    test_anomaly: Path
    gt_anomaly: Path | None


def recognize_real_iad_layout(root: Path) -> RealIADLayout:
    """Recognize a Real-IAD-like layout (best-effort).

    This converter intentionally supports a small set of common industrial layouts:
    - custom-like: train/normal, test/normal, test/anomaly, ground_truth/anomaly
    - mvtec-like: train/good, test/good, test/bad, ground_truth/bad
    """

    root = Path(root)

    # Custom-like.
    cand_train = root / "train" / "normal"
    cand_test_n = root / "test" / "normal"
    cand_test_a = root / "test" / "anomaly"
    cand_gt = root / "ground_truth" / "anomaly"
    if cand_train.exists() and cand_test_n.exists() and cand_test_a.exists():
        gt = cand_gt if cand_gt.exists() else None
        return RealIADLayout(
            train_normal=cand_train,
            test_normal=cand_test_n,
            test_anomaly=cand_test_a,
            gt_anomaly=gt,
        )

    # MVTec-like.
    cand_train = root / "train" / "good"
    cand_test_n = root / "test" / "good"
    cand_test_a = root / "test" / "bad"
    cand_gt = root / "ground_truth" / "bad"
    if cand_train.exists() and cand_test_n.exists() and cand_test_a.exists():
        gt = cand_gt if cand_gt.exists() else None
        return RealIADLayout(
            train_normal=cand_train,
            test_normal=cand_test_n,
            test_anomaly=cand_test_a,
            gt_anomaly=gt,
        )

    raise ValueError(
        "Unable to recognize a supported Real-IAD-style layout.\n"
        "Supported patterns include:\n"
        "- <root>/train/normal + <root>/test/normal + <root>/test/anomaly (+ ground_truth/anomaly)\n"
        "- <root>/train/good + <root>/test/good + <root>/test/bad (+ ground_truth/bad)\n"
        f"root={root}"
    )


def convert_real_iad_to_manifest(
    *,
    root: str | Path,
    out_path: str | Path,
    category: str = "real_iad",
    absolute_paths: bool = False,
    include_masks: bool = False,
) -> list[dict[str, Any]]:
    """Convert a Real-IAD-like dataset tree into a JSONL manifest (paths-first)."""

    root_path = Path(root)
    out_file = Path(out_path)
    out_dir = out_file.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    cat = str(category).strip() or "real_iad"
    layout = recognize_real_iad_layout(root_path)

    records: list[dict[str, Any]] = []
    for p in _iter_images(layout.train_normal):
        records.append(
            {
                "image_path": _relpath(p, base=out_dir, absolute=absolute_paths),
                "category": cat,
                "split": "train",
                "meta": (_extract_meta_best_effort(p) or None),
            }
        )

    for p in _iter_images(layout.test_normal):
        records.append(
            {
                "image_path": _relpath(p, base=out_dir, absolute=absolute_paths),
                "category": cat,
                "split": "test",
                "label": 0,
                "meta": (_extract_meta_best_effort(p) or None),
            }
        )

    gt_dir = layout.gt_anomaly
    for p in _iter_images(layout.test_anomaly):
        rec: dict[str, Any] = {
            "image_path": _relpath(p, base=out_dir, absolute=absolute_paths),
            "category": cat,
            "split": "test",
            "label": 1,
            "meta": (_extract_meta_best_effort(p) or None),
        }
        if include_masks and gt_dir is not None and gt_dir.exists():
            cand = gt_dir / f"{p.stem}_mask.png"
            if not cand.exists():
                cand = gt_dir / f"{p.stem}.png"
            if cand.exists() and cand.is_file():
                rec["mask_path"] = _relpath(cand, base=out_dir, absolute=absolute_paths)
        records.append(rec)

    # Stable order for reproducible diffs.
    records.sort(key=lambda r: (str(r.get("split", "")), str(r.get("image_path", ""))))

    with out_file.open("w", encoding="utf-8") as f:
        for rec in records:
            # Drop null meta for compactness.
            if rec.get("meta", None) in (None, {}):
                rec = dict(rec)
                rec.pop("meta", None)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return records


__all__ = ["RealIADLayout", "recognize_real_iad_layout", "convert_real_iad_to_manifest"]
