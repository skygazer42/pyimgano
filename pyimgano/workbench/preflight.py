from __future__ import annotations

from pyimgano.workbench.config import WorkbenchConfig
from pyimgano.workbench.preflight_issue_factory import build_preflight_issue
from pyimgano.workbench.preflight_model_compat import run_workbench_model_compat_preflight
from pyimgano.workbench.preflight_report import build_preflight_report
from pyimgano.workbench.preflight_summary import resolve_workbench_preflight_summary
from pyimgano.workbench.preflight_types import IssueSeverity, PreflightIssue, PreflightReport


def _fallback_dataset_readiness(issues: list[object]) -> dict[str, object]:
    issue_details: list[dict[str, str]] = []
    has_error = False
    for item in issues:
        if isinstance(item, dict):
            code = item.get("code")
            message = item.get("message")
            severity = item.get("severity")
        else:
            code = getattr(item, "code", None)
            message = getattr(item, "message", None)
            severity = getattr(item, "severity", None)
        if code is None or message is None:
            continue
        if str(severity) == "error":
            has_error = True
        issue_details.append(
            {
                "code": str(code),
                "message": str(message),
            }
        )
    return {
        "status": ("error" if has_error else "ok"),
        "issue_codes": [str(item["code"]) for item in issue_details],
        "issue_details": issue_details,
    }


def _build_workbench_dataset_readiness(
    *,
    config: WorkbenchConfig,
    issues: list[object],
) -> dict[str, object]:
    fallback = _fallback_dataset_readiness(issues)
    if str(fallback.get("status")) == "error":
        return fallback

    try:
        from pyimgano.datasets.inspection import profile_dataset_target

        dataset_name = str(config.dataset.name)
        category = str(config.dataset.category) if config.dataset.category is not None else None
        if dataset_name.lower() == "manifest":
            manifest_path = config.dataset.manifest_path
            if manifest_path is None:
                raise ValueError("manifest_path missing from config")
            payload = profile_dataset_target(
                target=str(manifest_path),
                dataset="manifest",
                category=category,
                root_fallback=str(config.dataset.root),
            )
        else:
            payload = profile_dataset_target(
                target=str(config.dataset.root),
                dataset=dataset_name,
                category=category,
            )
        readiness = payload.get("readiness", None)
        if isinstance(readiness, dict):
            return dict(readiness)
    except Exception:
        pass
    return fallback


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
    summary = dict(summary)
    summary["dataset_readiness"] = _build_workbench_dataset_readiness(
        config=config,
        issues=issues,
    )

    return build_preflight_report(
        dataset=dataset,
        category=category,
        summary=summary,
        issues=issues,
    )
