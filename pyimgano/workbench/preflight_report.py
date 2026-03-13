from __future__ import annotations

from typing import Any

from pyimgano.workbench.preflight_types import PreflightIssue, PreflightReport


def build_preflight_report(
    *,
    dataset: str,
    category: str,
    summary: dict[str, Any],
    issues: list[PreflightIssue],
) -> PreflightReport:
    return PreflightReport(
        dataset=str(dataset),
        category=str(category),
        summary=summary,
        issues=issues,
    )


__all__ = ["build_preflight_report"]
