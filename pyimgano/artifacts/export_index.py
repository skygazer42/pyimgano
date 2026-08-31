from __future__ import annotations

"""Versioned, content-addressed indexes for multi-artifact export roots."""

import hashlib
import json
import re
import unicodedata
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping, Sequence

from pyimgano.artifacts.manifest import canonical_json_bytes

EXPORT_INDEX_FILENAME = "export_index.json"
EXPORT_INDEX_SCHEMA_FAMILY = "pyimgano-export-index"
EXPORT_INDEX_SCHEMA_VERSION = 1

_IDENTITY_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_SAFE_SLUG_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,62}[A-Za-z0-9])?$")
_FORMATS = frozenset({"native", "onnx", "torchscript", "openvino"})
_BACKENDS = frozenset({"pyimgano", "onnxruntime", "torchscript", "openvino"})
_WINDOWS_RESERVED = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{index}" for index in range(1, 10)}
    | {f"LPT{index}" for index in range(1, 10)}
)


class ExportIndexError(ValueError):
    """Raised when an export index is ambiguous, unsafe, or has been modified."""


def _is_windows_reserved(value: str) -> bool:
    basename = value.rstrip(" .").split(".", 1)[0]
    return basename.upper() in _WINDOWS_RESERVED


def _identity(payload: Mapping[str, Any]) -> str:
    projection = dict(payload)
    projection.pop("index_id", None)
    return f"sha256:{hashlib.sha256(canonical_json_bytes(projection)).hexdigest()}"


def _normalized_category(value: Any, *, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ExportIndexError(f"{path}: expected a non-empty category string")
    normalized = unicodedata.normalize("NFC", value)
    if normalized != value:
        raise ExportIndexError(f"{path}: category must already use Unicode NFC normalization")
    if value != value.strip() or any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ExportIndexError(f"{path}: category contains unsafe whitespace/control characters")
    return value


def category_slug(category: str) -> str:
    """Return a deterministic cross-platform-safe directory name for a category."""

    normalized = _normalized_category(category, path="category")
    if (
        _SAFE_SLUG_RE.fullmatch(normalized)
        and not _is_windows_reserved(normalized)
        and not normalized.endswith((".", " "))
    ):
        return normalized

    folded = unicodedata.normalize("NFKD", normalized)
    ascii_text = folded.encode("ascii", "ignore").decode("ascii")
    base = re.sub(r"[^A-Za-z0-9._-]+", "-", ascii_text).strip(" .-_")
    if not base or _is_windows_reserved(base):
        base = "category"
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]
    room = 64 - len(digest) - 1
    base = base[:room].rstrip(" .-_") or "category"
    return f"{base}-{digest}"


def _relative_path(value: Any, *, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ExportIndexError(f"{path}: expected a non-empty relative path")
    if "\\" in value or "\x00" in value or "//" in value or value.endswith("/"):
        raise ExportIndexError(f"{path}: expected a normalized relative POSIX path")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if posix.is_absolute() or windows.is_absolute() or windows.drive:
        raise ExportIndexError(f"{path}: absolute or drive-qualified paths are forbidden")
    if any(part in {"", ".", ".."} for part in value.split("/")):
        raise ExportIndexError(f"{path}: dot, dot-dot, and empty path segments are forbidden")
    return value


def validate_export_index(
    payload: Mapping[str, Any],
    *,
    root: str | Path | None = None,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ExportIndexError("export index must be a JSON object")
    data = dict(payload)
    unknown = set(data) - {"schema_family", "schema_version", "index_id", "entries"}
    if unknown:
        raise ExportIndexError(f"export index contains unknown keys: {sorted(unknown)!r}")
    if data.get("schema_family") != EXPORT_INDEX_SCHEMA_FAMILY:
        raise ExportIndexError(f"schema_family: expected {EXPORT_INDEX_SCHEMA_FAMILY!r}")
    if data.get("schema_version") != EXPORT_INDEX_SCHEMA_VERSION:
        raise ExportIndexError(f"schema_version: expected {EXPORT_INDEX_SCHEMA_VERSION}")
    declared_id = data.get("index_id")
    if not isinstance(declared_id, str) or not _IDENTITY_RE.fullmatch(declared_id):
        raise ExportIndexError("index_id: expected sha256:<64 lowercase hex characters>")
    expected_id = _identity(data)
    if declared_id != expected_id:
        raise ExportIndexError(
            f"index_id: does not match canonical export index identity {expected_id}"
        )

    raw_entries = data.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ExportIndexError("entries: expected a non-empty array")
    if len(raw_entries) > 4096:
        raise ExportIndexError("entries: exceeds the schema-v1 limit")

    normalized_entries: list[dict[str, Any]] = []
    category_keys: dict[str, str] = {}
    category_slugs: dict[str, str] = {}
    selection_keys: set[tuple[str, str, str]] = set()
    artifact_paths: set[str] = set()
    artifact_ids: set[str] = set()
    resolved_root = Path(root).resolve() if root is not None else None
    for index, raw in enumerate(raw_entries):
        field = f"entries[{index}]"
        if not isinstance(raw, Mapping):
            raise ExportIndexError(f"{field}: expected a JSON object")
        item = dict(raw)
        unknown_item = set(item) - {
            "category",
            "slug",
            "format",
            "backend",
            "artifact",
            "manifest",
            "artifact_id",
        }
        if unknown_item:
            raise ExportIndexError(f"{field}: unknown keys: {sorted(unknown_item)!r}")
        category = _normalized_category(item.get("category"), path=f"{field}.category")
        slug = item.get("slug")
        if not isinstance(slug, str) or slug != category_slug(category):
            raise ExportIndexError(
                f"{field}.slug: must equal the canonical category slug {category_slug(category)!r}"
            )
        case_key = unicodedata.normalize("NFC", category).casefold()
        previous_category = category_keys.setdefault(case_key, category)
        if previous_category != category:
            raise ExportIndexError(
                "entries: categories collide after Unicode case normalization: "
                f"{previous_category!r} and {category!r}"
            )
        previous_slug = category_slugs.setdefault(slug.casefold(), category)
        if previous_slug != category:
            raise ExportIndexError(
                f"entries: category slugs collide: {previous_slug!r} and {category!r}"
            )

        artifact_format = str(item.get("format", ""))
        backend = str(item.get("backend", ""))
        if artifact_format not in _FORMATS:
            raise ExportIndexError(f"{field}.format: unsupported format {artifact_format!r}")
        if backend not in _BACKENDS:
            raise ExportIndexError(f"{field}.backend: unsupported backend {backend!r}")
        selection_key = (case_key, artifact_format, backend)
        if selection_key in selection_keys:
            raise ExportIndexError(
                f"{field}: duplicate category/format/backend selection {selection_key!r}"
            )
        selection_keys.add(selection_key)

        artifact = _relative_path(item.get("artifact"), path=f"{field}.artifact")
        manifest = _relative_path(item.get("manifest"), path=f"{field}.manifest")
        expected_manifest = f"{artifact}/artifact_manifest.json"
        if manifest != expected_manifest:
            raise ExportIndexError(
                f"{field}.manifest: expected {expected_manifest!r} for the artifact root"
            )
        if artifact in artifact_paths:
            raise ExportIndexError(f"{field}.artifact: duplicate artifact reference")
        artifact_paths.add(artifact)
        artifact_id = item.get("artifact_id")
        if not isinstance(artifact_id, str) or not _IDENTITY_RE.fullmatch(artifact_id):
            raise ExportIndexError(
                f"{field}.artifact_id: expected sha256:<64 lowercase hex characters>"
            )
        if artifact_id in artifact_ids:
            raise ExportIndexError(f"{field}.artifact_id: duplicate artifact identity")
        artifact_ids.add(artifact_id)
        if resolved_root is not None:
            target = (resolved_root / PurePosixPath(manifest)).resolve()
            try:
                target.relative_to(resolved_root)
            except ValueError as exc:
                raise ExportIndexError(f"{field}.manifest: escapes the export root") from exc
            if not target.is_file():
                raise ExportIndexError(f"{field}.manifest: referenced file is missing")
        normalized_entries.append(
            {
                "category": category,
                "slug": slug,
                "format": artifact_format,
                "backend": backend,
                "artifact": artifact,
                "manifest": manifest,
                "artifact_id": artifact_id,
            }
        )
    data["entries"] = normalized_entries
    return data


def build_export_index(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    normalized_entries = [dict(item) for item in entries]
    normalized_entries.sort(
        key=lambda item: (
            str(item.get("category", "")).casefold(),
            str(item.get("format", "")),
            str(item.get("backend", "")),
        )
    )
    payload: dict[str, Any] = {
        "schema_family": EXPORT_INDEX_SCHEMA_FAMILY,
        "schema_version": EXPORT_INDEX_SCHEMA_VERSION,
        "entries": normalized_entries,
    }
    payload["index_id"] = _identity(payload)
    return validate_export_index(payload)


def load_export_index(
    path: str | Path,
    *,
    root: str | Path | None = None,
) -> dict[str, Any]:
    source = Path(path)
    try:
        raw = source.read_bytes()
    except OSError as exc:
        raise ExportIndexError(f"Failed to read export index {source}: {exc}") from exc
    if len(raw) > 4 * 1024 * 1024:
        raise ExportIndexError("export index exceeds the schema-v1 byte limit")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExportIndexError(f"Invalid export index JSON {source}: {exc}") from exc
    return validate_export_index(payload, root=(root if root is not None else source.parent))


def write_export_index(path: str | Path, entries: Sequence[Mapping[str, Any]]) -> Path:
    destination = Path(path)
    payload = build_export_index(entries)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(canonical_json_bytes(payload) + b"\n")
    return destination


__all__ = [
    "EXPORT_INDEX_FILENAME",
    "EXPORT_INDEX_SCHEMA_FAMILY",
    "EXPORT_INDEX_SCHEMA_VERSION",
    "ExportIndexError",
    "build_export_index",
    "category_slug",
    "load_export_index",
    "validate_export_index",
    "write_export_index",
]
