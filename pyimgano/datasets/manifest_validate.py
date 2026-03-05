from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from pyimgano.datasets.manifest import ManifestRecord, _resolve_existing_path


@dataclass(frozen=True)
class ManifestValidationReport:
    manifest_path: str
    category: str | None
    record_count: int
    categories: list[str]
    ok: bool
    errors: list[str]
    warnings: list[str]

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "manifest_path": str(self.manifest_path),
            "category": (str(self.category) if self.category is not None else None),
            "record_count": int(self.record_count),
            "categories": list(self.categories),
            "ok": bool(self.ok),
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


def _iter_raw_lines(path: Path) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            if text.startswith("#"):
                continue
            out.append((int(lineno), text))
    return out


def validate_manifest_file(
    *,
    manifest_path: str | Path,
    root_fallback: str | Path | None,
    check_files: bool = True,
    category: str | None = None,
) -> ManifestValidationReport:
    """Validate a JSONL manifest file.

    Validation levels:
    - Schema: required keys + split/label consistency via `ManifestRecord.from_mapping`.
    - Files (optional): image/mask paths exist on disk (best-effort, respects root fallback).
    """

    mp = Path(manifest_path)
    errors: list[str] = []
    warnings: list[str] = []

    if not mp.exists():
        return ManifestValidationReport(
            manifest_path=str(mp),
            category=(str(category) if category is not None else None),
            record_count=0,
            categories=[],
            ok=False,
            errors=[f"Manifest not found: {mp}"],
            warnings=[],
        )
    if not mp.is_file():
        return ManifestValidationReport(
            manifest_path=str(mp),
            category=(str(category) if category is not None else None),
            record_count=0,
            categories=[],
            ok=False,
            errors=[f"Manifest path is not a file: {mp}"],
            warnings=[],
        )

    root_path = None if root_fallback is None else Path(root_fallback)

    records: list[ManifestRecord] = []
    cats: set[str] = set()

    for lineno, text in _iter_raw_lines(mp):
        raw: Mapping[str, Any]
        try:
            loaded = json.loads(text)
        except json.JSONDecodeError as exc:
            errors.append(f"Manifest line {lineno}: invalid JSON ({exc.msg}).")
            continue
        if not isinstance(loaded, Mapping):
            errors.append(
                f"Manifest line {lineno}: expected a JSON object, got {type(loaded).__name__}."
            )
            continue
        raw = loaded
        try:
            rec = ManifestRecord.from_mapping(raw, lineno=int(lineno))
        except Exception as exc:  # noqa: BLE001 - validation boundary
            errors.append(str(exc))
            continue

        cats.add(str(rec.category))
        records.append(rec)

    category_norm = str(category).strip() if category is not None else None
    if category_norm is not None and not category_norm:
        category_norm = None

    if category_norm is not None:
        records_checked = [r for r in records if str(r.category) == category_norm]
        if not records_checked and records:
            warnings.append(
                f"Manifest contains no records for category={category_norm!r}. "
                f"Available: {sorted(cats)}"
            )
    else:
        records_checked = list(records)

    if check_files:
        for rec in records_checked:
            try:
                _resolve_existing_path(rec.image_path, manifest_path=mp, root_fallback=root_path)
            except Exception as exc:  # noqa: BLE001 - best-effort validation
                errors.append(str(exc))

            if rec.mask_path is not None:
                try:
                    _resolve_existing_path(rec.mask_path, manifest_path=mp, root_fallback=root_path)
                except Exception as exc:  # noqa: BLE001 - best-effort validation
                    errors.append(str(exc))
            elif rec.label == 1:
                warnings.append(
                    "Manifest record with label=1 has no mask_path; pixel metrics may be disabled."
                )

    ok = not errors
    return ManifestValidationReport(
        manifest_path=str(mp),
        category=category_norm,
        record_count=int(len(records)),
        categories=sorted(cats),
        ok=bool(ok),
        errors=errors,
        warnings=warnings,
    )


__all__ = ["ManifestValidationReport", "validate_manifest_file"]
