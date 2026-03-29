from __future__ import annotations

from typing import Any, Mapping


def resolve_non_manifest_preflight_source_or_summary(
    source: Mapping[str, Any],
) -> dict[str, Any] | None:
    summary = source.get("summary", None)
    return dict(summary) if isinstance(summary, Mapping) else None


def resolve_non_manifest_category_listing_summary(
    category_listing: Mapping[str, Any],
) -> dict[str, Any] | None:
    summary = category_listing.get("summary", None)
    return dict(summary) if isinstance(summary, Mapping) else None


__all__ = [
    "resolve_non_manifest_category_listing_summary",
    "resolve_non_manifest_preflight_source_or_summary",
]
