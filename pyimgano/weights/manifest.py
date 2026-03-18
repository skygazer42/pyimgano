from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from pyimgano.utils.security import FileHasher


@dataclass(frozen=True)
class WeightsManifestReport:
    ok: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    entries: tuple[dict[str, Any], ...]

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "ok": bool(self.ok),
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "entries": list(self.entries),
        }


def _as_dict(obj: Any) -> dict[str, Any] | None:
    return obj if isinstance(obj, dict) else None


def _as_list(obj: Any) -> list[Any] | None:
    return obj if isinstance(obj, list) else None


def _resolve_path(raw: str, *, base_dir: Path | None) -> Path:
    p = Path(str(raw))
    if p.is_absolute():
        return p
    if base_dir is None:
        return p
    return (base_dir / p).resolve()


def validate_weights_manifest(
    manifest: Mapping[str, Any],
    *,
    base_dir: str | Path | None = None,
    check_files: bool = True,
    check_hashes: bool = False,
) -> WeightsManifestReport:
    """Validate a weights manifest payload.

    This is an intentionally lightweight "model weights inventory" helper.
    It does not download anything; it only validates schema and local files.

    Schema (v1-ish; best-effort):
    - top-level: {"schema_version": 1, "entries": [ ... ]}
    - each entry: {"name": str, "path": str, "sha256": str?}
    """

    errors: list[str] = []
    warnings: list[str] = []

    base = Path(base_dir).resolve() if base_dir is not None else None

    version = manifest.get("schema_version", None)
    if version is None:
        warnings.append("Missing top-level key: schema_version (recommended).")
    else:
        try:
            v = int(version)
            if v != 1:
                warnings.append(f"Unknown schema_version={v}; validation is best-effort.")
        except Exception:
            warnings.append(f"Invalid schema_version={version!r}; expected int.")

    entries_any = manifest.get("entries", None)
    entries_list = _as_list(entries_any)
    if entries_list is None:
        errors.append("Missing or invalid top-level key: entries (expected list).")
        return WeightsManifestReport(
            ok=False, errors=tuple(errors), warnings=tuple(warnings), entries=()
        )

    seen_names: set[str] = set()
    seen_paths: set[str] = set()
    normalized: list[dict[str, Any]] = []

    for i, raw in enumerate(entries_list):
        entry = _as_dict(raw)
        if entry is None:
            errors.append(f"entries[{i}] must be an object (dict).")
            continue

        name_any = entry.get("name", None)
        path_any = entry.get("path", None)
        if name_any is None or not str(name_any).strip():
            errors.append(f"entries[{i}] missing required key: name")
            continue
        if path_any is None or not str(path_any).strip():
            errors.append(f"entries[{i}] missing required key: path")
            continue

        name = str(name_any).strip()
        raw_path = str(path_any).strip()

        if name in seen_names:
            errors.append(f"Duplicate entry name: {name!r}")
            continue
        seen_names.add(name)

        abs_path = _resolve_path(raw_path, base_dir=base)
        abs_path_str = str(abs_path)
        if abs_path_str in seen_paths:
            warnings.append(f"Duplicate entry path: {raw_path!r} (resolved: {abs_path})")
        seen_paths.add(abs_path_str)

        sha_any = entry.get("sha256", None)
        sha = str(sha_any).strip().lower() if sha_any is not None else None
        if sha == "":
            sha = None

        if _nonempty_str(entry.get("source", None)) is None:
            warnings.append(f"Entry {name!r} is missing recommended metadata: source")
        if _nonempty_str(entry.get("license", None)) is None:
            warnings.append(f"Entry {name!r} is missing recommended metadata: license")
        runtime = _nonempty_str(entry.get("runtime", None))
        runtimes = entry.get("runtimes", None)
        if runtime is None and not _has_nonempty_string_list(runtimes):
            warnings.append(f"Entry {name!r} is missing recommended metadata: runtime")

        if check_files and not abs_path.exists():
            errors.append(f"Missing weights file for {name!r}: {abs_path}")
        if check_hashes and sha is not None and abs_path.exists():
            try:
                actual = FileHasher.compute_hash(str(abs_path), algorithm="sha256")
                if actual.lower() != sha.lower():
                    errors.append(f"SHA256 mismatch for {name!r}: expected={sha} actual={actual}")
            except Exception as exc:  # noqa: BLE001 - best-effort validation
                errors.append(f"Failed to compute sha256 for {name!r}: {exc}")

        # Keep normalized entries stable and JSON-friendly.
        norm = dict(entry)
        norm["name"] = name
        norm["path"] = raw_path
        norm["resolved_path"] = abs_path_str
        if sha is not None:
            norm["sha256"] = sha
        normalized.append(norm)

    ok = len(errors) == 0
    return WeightsManifestReport(
        ok=bool(ok),
        errors=tuple(errors),
        warnings=tuple(warnings),
        entries=tuple(normalized),
    )


def _nonempty_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _has_nonempty_string_list(value: Any) -> bool:
    items = _as_list(value)
    if items is None:
        return False
    return any(_nonempty_str(item) is not None for item in items)


def load_weights_manifest_file(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Weights manifest not found: {p}")
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Failed to parse weights manifest JSON: {p}") from exc
    if not isinstance(obj, dict):
        raise ValueError(f"Weights manifest must be a JSON object (dict): {p}")
    return obj


def validate_weights_manifest_file(
    *,
    manifest_path: str | Path,
    base_dir: str | Path | None = None,
    check_files: bool = True,
    check_hashes: bool = False,
) -> WeightsManifestReport:
    payload = load_weights_manifest_file(manifest_path)
    base = Path(base_dir) if base_dir is not None else Path(manifest_path).resolve().parent
    return validate_weights_manifest(
        payload,
        base_dir=base,
        check_files=bool(check_files),
        check_hashes=bool(check_hashes),
    )


__all__ = [
    "WeightsManifestReport",
    "load_weights_manifest_file",
    "validate_weights_manifest",
    "validate_weights_manifest_file",
]
