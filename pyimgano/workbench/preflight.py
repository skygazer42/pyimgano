from __future__ import annotations

from pyimgano.workbench.config import WorkbenchConfig
from pyimgano.workbench.preflight_issue_factory import build_preflight_issue
from pyimgano.workbench.preflight_model_compat import run_workbench_model_compat_preflight
from pyimgano.workbench.preflight_report import build_preflight_report
from pyimgano.workbench.preflight_summary import resolve_workbench_preflight_summary
from pyimgano.workbench.preflight_types import (
    IssueSeverity,
    PreflightIssue,
    PreflightReport,
)


def run_preflight(*, config: WorkbenchConfig) -> PreflightReport:
    """Run best-effort dataset validation and return a JSON-friendly report."""

    _ = IssueSeverity
    dataset = str(config.dataset.name)
    category = str(config.dataset.category)

    issues: list[PreflightIssue] = []
    run_workbench_model_compat_preflight(
        config=config,
        issues=issues,
        issue_builder=build_preflight_issue,
    )
    summary = resolve_workbench_preflight_summary(
        config=config,
        issues=issues,
        issue_builder=build_preflight_issue,
    )

    return build_preflight_report(
        dataset=dataset,
        category=category,
        summary=summary,
        issues=issues,
    )
