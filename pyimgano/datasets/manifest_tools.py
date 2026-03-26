from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any


def iter_manifest_rows(manifest_path: str | Path) -> Iterator[dict[str, Any]]:
    """Iterate JSONL manifest rows as dicts, preserving unknown keys.

    Notes
    -----
    - Ignores empty lines and comment lines starting with '#'.
    - Requires `image_path` and `category` to be present and non-empty.
    """

    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            if text.startswith("#"):
                continue
            try:
                raw = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Manifest line {i}: invalid JSON ({exc.msg}).") from exc
            if not isinstance(raw, Mapping):
                raise ValueError(
                    f"Manifest line {i}: expected a JSON object, got {type(raw).__name__}."
                )
            row = dict(raw)

            image_path = str(row.get("image_path", "")).strip()
            category = str(row.get("category", "")).strip()
            if not image_path:
                raise ValueError(f"Manifest line {i}: missing required field 'image_path'.")
            if not category:
                raise ValueError(f"Manifest line {i}: missing required field 'category'.")

            yield row


def _normalize_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    return s or None


def _normalize_split(value: Any) -> str | None:
    s = _normalize_optional_str(value)
    return None if s is None else s.lower()


def _normalize_label(value: Any) -> int | None:
    s = _normalize_optional_str(value)
    if s is None:
        return None
    try:
        return int(s)
    except Exception:
        return None


def _path_exists(
    raw_value: Any,
    *,
    manifest_dir: Path,
    root_fallback: Path | None,
) -> bool:
    raw = _normalize_optional_str(raw_value)
    if raw is None:
        return False
    p = Path(raw)
    if p.is_absolute():
        return p.exists()
    cand1 = (manifest_dir / p).resolve()
    if cand1.exists():
        return True
    if root_fallback is not None:
        cand2 = (root_fallback / p).resolve()
        return cand2.exists()
    return False


def manifest_stats(
    *,
    manifest_path: str | Path,
    root_fallback: str | Path | None = None,
) -> dict[str, Any]:
    """Compute basic statistics for a JSONL manifest."""

    mp = Path(manifest_path)
    manifest_dir = mp.resolve().parent
    root_path = None if root_fallback is None else Path(root_fallback)

    total = 0
    cat_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    mask_present = 0
    image_abs = 0

    image_exists = 0
    mask_exists = 0

    for row in iter_manifest_rows(mp):
        total += 1
        cat = str(row.get("category", "")).strip()
        cat_counts[cat] += 1

        split = _normalize_split(row.get("split", None))
        split_counts[split or "missing"] += 1

        label = _normalize_label(row.get("label", None))
        if label is None:
            label_counts["missing"] += 1
        else:
            label_counts[str(label)] += 1

        mask_path = _normalize_optional_str(row.get("mask_path", None))
        if mask_path is not None:
            mask_present += 1
            if root_path is not None and _path_exists(
                mask_path, manifest_dir=manifest_dir, root_fallback=root_path
            ):
                mask_exists += 1

        image_path = str(row.get("image_path", "")).strip()
        if Path(image_path).is_absolute():
            image_abs += 1

        if root_path is not None and _path_exists(
            image_path, manifest_dir=manifest_dir, root_fallback=root_path
        ):
            image_exists += 1

    denom = float(total) if total else 1.0
    payload: dict[str, Any] = {
        "total_records": int(total),
        "category_counts": dict(sorted(cat_counts.items(), key=lambda kv: kv[0])),
        "split_counts": dict(sorted(split_counts.items(), key=lambda kv: kv[0])),
        "label_counts": dict(sorted(label_counts.items(), key=lambda kv: kv[0])),
        "mask_path_present_count": int(mask_present),
        "mask_path_present_ratio": float(mask_present / denom),
        "image_path_absolute_count": int(image_abs),
        "image_path_absolute_ratio": float(image_abs / denom),
    }

    if root_path is not None:
        present_denom = float(mask_present) if mask_present else 1.0
        payload["file_checks"] = {
            "enabled": True,
            "root_fallback": str(root_path),
            "image_exists_count": int(image_exists),
            "image_exists_ratio": float(image_exists / denom),
            "mask_exists_count": int(mask_exists),
            "mask_exists_ratio_of_present": float(mask_exists / present_denom),
        }

    return payload


def filter_manifest_records(
    *,
    manifest_path: str | Path,
    category: str | None = None,
    split: str | None = None,
    label: int | None = None,
    has_mask: bool | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Filter manifest rows by simple field predicates.

    Preserves the original record objects (unknown keys included) and keeps the
    input order stable.
    """

    cat_f = None if category is None else str(category).strip()
    split_f = None if split is None else str(split).strip().lower()
    label_f = None if label is None else int(label)
    limit_f = None if limit is None else int(limit)

    if limit_f is not None and limit_f < 0:
        raise ValueError("--limit must be >= 0")

    out: list[dict[str, Any]] = []
    for row in iter_manifest_rows(manifest_path):
        if cat_f is not None and str(row.get("category", "")).strip() != cat_f:
            continue

        if split_f is not None:
            s_norm = _normalize_split(row.get("split", None))
            if s_norm != split_f:
                continue

        if label_f is not None:
            lbl = _normalize_label(row.get("label", None))
            if lbl is None or int(lbl) != int(label_f):
                continue

        if has_mask is not None:
            present = _normalize_optional_str(row.get("mask_path", None)) is not None
            if bool(has_mask) != bool(present):
                continue

        out.append(row)
        if limit_f is not None and len(out) >= limit_f:
            break

    return out


def write_manifest_records(*, records: list[Mapping[str, Any]], out_path: str | Path) -> None:
    """Write JSONL manifest records to disk."""

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(dict(rec), ensure_ascii=False) + "\n")


__all__ = [
    "filter_manifest_records",
    "iter_manifest_rows",
    "manifest_stats",
    "write_manifest_records",
]
