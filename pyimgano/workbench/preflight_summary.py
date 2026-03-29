from __future__ import annotations

from typing import Any, Callable

from pyimgano.workbench.config import WorkbenchConfig
from pyimgano.workbench.manifest_preflight import run_manifest_preflight
from pyimgano.workbench.non_manifest_preflight import run_non_manifest_preflight
from pyimgano.workbench.preflight_dispatch import resolve_preflight_dataset_dispatch


def resolve_workbench_preflight_summary(
    *,
    config: WorkbenchConfig,
    issues: list[Any],
    issue_builder: Callable[..., Any],
) -> dict[str, Any]:
    dataset_dispatch = resolve_preflight_dataset_dispatch(config=config)
    if dataset_dispatch == "manifest":
        return run_manifest_preflight(config=config, issues=issues, issue_builder=issue_builder)
    return run_non_manifest_preflight(config=config, issues=issues, issue_builder=issue_builder)


__all__ = ["resolve_workbench_preflight_summary"]
