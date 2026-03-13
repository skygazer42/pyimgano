from __future__ import annotations

from typing import Any, Mapping

from pyimgano.reporting.report import save_run_report
from pyimgano.reporting.runs import WorkbenchRunPaths


def persist_workbench_run_report(
    *,
    payload: Mapping[str, Any],
    paths: WorkbenchRunPaths | None,
) -> dict[str, Any]:
    out = dict(payload)
    if paths is None:
        return out

    out["run_dir"] = str(paths.run_dir)
    save_run_report(paths.report_json, out)
    return out


__all__ = ["persist_workbench_run_report"]
