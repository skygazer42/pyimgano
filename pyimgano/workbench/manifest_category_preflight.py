from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

from pyimgano.workbench.manifest_category_assignment import analyze_manifest_category_assignment
from pyimgano.workbench.manifest_category_paths import inspect_manifest_category_paths
from pyimgano.workbench.manifest_category_summary import summarize_manifest_category_records


def preflight_manifest_category(
    *,
    category: str,
    records: Sequence[Any],
    manifest_path: Path,
    root_fallback: Path | None,
    policy: Any,
    issues: list[Any],
    issue_builder: Callable[..., Any],
) -> dict[str, Any]:
    category_summary = summarize_manifest_category_records(records=records)
    recs = category_summary["records"]

    mask_exists_by_index = inspect_manifest_category_paths(
        category=category,
        records=recs,
        manifest_path=manifest_path,
        root_fallback=root_fallback,
        issues=issues,
        issue_builder=issue_builder,
    )

    assignment_summary = analyze_manifest_category_assignment(
        category=category,
        records=recs,
        policy=policy,
        mask_exists_by_index=mask_exists_by_index,
        issues=issues,
        issue_builder=issue_builder,
    )

    return {
        "counts": category_summary["counts"],
        "assigned_counts": assignment_summary["assigned_counts"],
        "mask_coverage": assignment_summary["mask_coverage"],
        "pixel_metrics": assignment_summary["pixel_metrics"],
    }


__all__ = ["preflight_manifest_category"]
