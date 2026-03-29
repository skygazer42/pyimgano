from __future__ import annotations

from typing import Any, Callable

from pyimgano.workbench.config import WorkbenchConfig
from pyimgano.workbench.manifest_category_selection import select_manifest_preflight_categories
from pyimgano.workbench.manifest_preflight_categories import preflight_manifest_categories
from pyimgano.workbench.manifest_preflight_flow import (
    resolve_manifest_preflight_source_or_summary,
    resolve_manifest_record_preflight_summary,
)
from pyimgano.workbench.manifest_preflight_report import build_manifest_preflight_report
from pyimgano.workbench.manifest_record_preflight import resolve_manifest_preflight_records
from pyimgano.workbench.manifest_source_validation import resolve_manifest_preflight_source
from pyimgano.workbench.manifest_split_policy import build_manifest_split_policy


def run_manifest_preflight(
    *,
    config: WorkbenchConfig,
    issues: list[Any],
    issue_builder: Callable[..., Any],
) -> dict[str, Any]:
    source = resolve_manifest_preflight_source(
        config=config,
        issues=issues,
        issue_builder=issue_builder,
    )
    source_summary = resolve_manifest_preflight_source_or_summary(source)
    if source_summary is not None:
        return source_summary

    mp = source["manifest_path"]
    root_fallback = source["root_fallback"]

    policy = build_manifest_split_policy(config=config)

    record_preflight = resolve_manifest_preflight_records(
        manifest_path=mp,
        issues=issues,
        issue_builder=issue_builder,
    )
    record_summary = resolve_manifest_record_preflight_summary(record_preflight)
    if record_summary is not None:
        return record_summary
    records = record_preflight["records"]
    raw_categories = record_preflight["categories"]

    selected_categories = select_manifest_preflight_categories(
        requested_category=str(config.dataset.category),
        available_categories=raw_categories,
        issues=issues,
        issue_builder=issue_builder,
    )
    requested_all = bool(selected_categories["requested_all"])
    categories = selected_categories["categories"]

    per_category = preflight_manifest_categories(
        categories=categories,
        records=records,
        manifest_path=mp,
        root_fallback=root_fallback,
        policy=policy,
        issues=issues,
        issue_builder=issue_builder,
    )

    return build_manifest_preflight_report(
        manifest_path=str(mp),
        root_fallback=(str(root_fallback) if root_fallback is not None else None),
        policy=policy,
        categories=categories,
        per_category=per_category,
        requested_all=requested_all,
    )


__all__ = ["run_manifest_preflight"]
