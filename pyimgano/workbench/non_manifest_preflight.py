from __future__ import annotations

from typing import Any, Callable

from pyimgano.workbench.config import WorkbenchConfig
from pyimgano.workbench.non_manifest_category_listing import (
    load_non_manifest_preflight_categories,
)
from pyimgano.workbench.non_manifest_category_selection import (
    select_non_manifest_preflight_categories,
)
from pyimgano.workbench.non_manifest_preflight_report import (
    build_non_manifest_preflight_report,
)
from pyimgano.workbench.non_manifest_source_validation import resolve_non_manifest_preflight_source


def run_non_manifest_preflight(
    *,
    config: WorkbenchConfig,
    issues: list[Any],
    issue_builder: Callable[..., Any],
) -> dict[str, Any]:
    source = resolve_non_manifest_preflight_source(
        config=config,
        issues=issues,
        issue_builder=issue_builder,
    )
    if source["summary"] is not None:
        return source["summary"]

    dataset = str(source["dataset"])
    root = source["root"]
    category = str(config.dataset.category)

    category_listing = load_non_manifest_preflight_categories(
        config=config,
        dataset=dataset,
        root=str(root),
        issues=issues,
        issue_builder=issue_builder,
    )
    if category_listing["summary"] is not None:
        return category_listing["summary"]
    categories = category_listing["categories"]

    select_non_manifest_preflight_categories(
        requested_category=category,
        available_categories=categories,
        dataset=dataset,
        root=str(root),
        issues=issues,
        issue_builder=issue_builder,
    )

    return build_non_manifest_preflight_report(root=str(root), categories=categories)


__all__ = ["run_non_manifest_preflight"]
