from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping


def resolve_manifest_preflight_records(
    *,
    manifest_path: Path,
    issues: list[Any],
    issue_builder: Callable[..., Any],
) -> dict[str, Any]:
    records, raw_categories = load_manifest_records_best_effort(
        manifest_path=manifest_path,
        issues=issues,
        issue_builder=issue_builder,
    )
    if not records:
        issues.append(
            issue_builder(
                "MANIFEST_EMPTY",
                "error",
                "Manifest contains no valid records.",
                context={"manifest_path": str(manifest_path)},
            )
        )
        return {
            "records": records,
            "categories": raw_categories,
            "summary": {"manifest_path": str(manifest_path), "manifest": {"ok": False}},
        }

    return {
        "records": records,
        "categories": raw_categories,
        "summary": None,
    }


def load_manifest_records_best_effort(
    *,
    manifest_path: Path,
    issues: list[Any],
    issue_builder: Callable[..., Any],
):
    from pyimgano.datasets.manifest import ManifestRecord

    records: list[ManifestRecord] = []
    raw_categories: set[str] = set()

    with manifest_path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            if text.startswith("#"):
                continue

            try:
                raw = _parse_manifest_json(text, lineno=lineno)
            except Exception as exc:  # noqa: BLE001 - preflight boundary
                issues.append(
                    issue_builder(
                        "MANIFEST_INVALID_JSON",
                        "error",
                        "Invalid JSON line in manifest.",
                        context={"lineno": int(lineno), "error": str(exc)},
                    )
                )
                continue

            try:
                rec = ManifestRecord.from_mapping(raw, lineno=lineno)
            except Exception as exc:  # noqa: BLE001 - validation boundary
                issues.append(
                    issue_builder(
                        "MANIFEST_INVALID_RECORD",
                        "error",
                        "Invalid manifest record.",
                        context={"lineno": int(lineno), "error": str(exc)},
                    )
                )
                continue

            records.append(rec)
            raw_categories.add(str(rec.category))

    return records, raw_categories


def _parse_manifest_json(text: str, *, lineno: int) -> Mapping[str, Any]:
    import json

    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Manifest line {lineno}: invalid JSON ({exc.msg}).") from exc
    if not isinstance(raw, Mapping):
        raise ValueError(f"Manifest line {lineno}: expected JSON object, got {type(raw).__name__}.")
    return raw


def resolve_manifest_path_best_effort(
    raw_value: str,
    *,
    manifest_path: Path,
    root_fallback: Path | None,
) -> tuple[str, bool, str]:
    raw = str(raw_value)
    p = Path(raw)
    if p.is_absolute():
        resolved = p.resolve()
        return str(resolved), bool(resolved.exists()), "absolute"

    cand1 = (manifest_path.parent / p).resolve()
    if cand1.exists():
        return str(cand1), True, "manifest_dir"

    if root_fallback is not None:
        cand2 = (root_fallback / p).resolve()
        if cand2.exists():
            return str(cand2), True, "root_fallback"

    return str(cand1), False, "manifest_dir"


__all__ = [
    "load_manifest_records_best_effort",
    "resolve_manifest_preflight_records",
    "resolve_manifest_path_best_effort",
]
