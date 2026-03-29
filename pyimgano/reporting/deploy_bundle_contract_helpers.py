from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


def collect_existing_artifact_refs(
    root: Path,
    *,
    paths: Mapping[str, str],
) -> dict[str, str]:
    refs: dict[str, str] = {}
    for name, rel_path in paths.items():
        if (root / rel_path).is_file():
            refs[str(name)] = str(rel_path)
    return refs


def build_artifact_roles(entries: list[dict[str, Any]]) -> dict[str, list[str]]:
    roles: dict[str, list[str]] = {}
    for entry in entries:
        rel_path = entry.get("path", None)
        if not isinstance(rel_path, str) or not rel_path.strip():
            continue
        role = entry.get("role", None)
        if not isinstance(role, str) or not role.strip():
            continue
        roles.setdefault(str(role), []).append(str(rel_path))
    return {
        str(role): sorted(paths)
        for role, paths in sorted(roles.items(), key=lambda item: str(item[0]))
    }


def build_artifact_digests(entries: list[dict[str, Any]]) -> dict[str, str]:
    digests: dict[str, str] = {}
    for entry in entries:
        rel_path = entry.get("path", None)
        sha256 = entry.get("sha256", None)
        if not isinstance(rel_path, str) or not rel_path.strip():
            continue
        if not isinstance(sha256, str) or not sha256.strip():
            continue
        digests[str(rel_path)] = str(sha256)
    return dict(sorted(digests.items(), key=lambda item: item[0]))


def required_artifacts_present(
    refs: Mapping[str, Any],
    *,
    required_names: tuple[str, ...],
) -> bool:
    return all(str(name) in refs for name in required_names)


__all__ = [
    "build_artifact_digests",
    "build_artifact_roles",
    "collect_existing_artifact_refs",
    "required_artifacts_present",
]
