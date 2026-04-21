from __future__ import annotations

from typing import Any, Mapping


def resolve_manifest_preflight_source_or_summary(
    source: Mapping[str, Any],
) -> dict[str, Any] | None:
    summary = source.get("summary", None)
    return dict(summary) if isinstance(summary, Mapping) else None


def resolve_manifest_record_preflight_summary(
    record_preflight: Mapping[str, Any],
) -> dict[str, Any] | None:
    summary = record_preflight.get("summary", None)
    return dict(summary) if isinstance(summary, Mapping) else None


__all__ = [
    "resolve_manifest_preflight_source_or_summary",
    "resolve_manifest_record_preflight_summary",
]
