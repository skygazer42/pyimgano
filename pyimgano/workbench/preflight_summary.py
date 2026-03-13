from __future__ import annotations

from typing import Any, Callable

from pyimgano.workbench.config import WorkbenchConfig
from pyimgano.workbench.manifest_preflight import run_manifest_preflight
from pyimgano.workbench.non_manifest_preflight import run_non_manifest_preflight


def resolve_workbench_preflight_summary(
    *,
    config: WorkbenchConfig,
    issues: list[Any],
    issue_builder: Callable[..., Any],
) -> dict[str, Any]:
    dataset = str(config.dataset.name)
    if dataset.lower() == "manifest":
        return run_manifest_preflight(config=config, issues=issues, issue_builder=issue_builder)
    return run_non_manifest_preflight(config=config, issues=issues, issue_builder=issue_builder)


__all__ = ["resolve_workbench_preflight_summary"]
