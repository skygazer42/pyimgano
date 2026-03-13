from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

from pyimgano.workbench.manifest_category_preflight import preflight_manifest_category


def preflight_manifest_categories(
    *,
    categories: Sequence[str],
    records: Sequence[Any],
    manifest_path: Path,
    root_fallback: Path | None,
    policy: Any,
    issues: list[Any],
    issue_builder: Callable[..., Any],
) -> dict[str, Any]:
    per_category: dict[str, Any] = {}
    for cat in categories:
        cat_records = [record for record in records if str(record.category) == str(cat)]
        per_category[str(cat)] = preflight_manifest_category(
            category=str(cat),
            records=cat_records,
            manifest_path=manifest_path,
            root_fallback=root_fallback,
            policy=policy,
            issues=issues,
            issue_builder=issue_builder,
        )
    return per_category


__all__ = ["preflight_manifest_categories"]
