from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


def load_workbench_config_from_run(run_dir: str | Path) -> Any:
    import pyimgano.workbench.load_run as load_run

    return load_run.load_workbench_config_from_run(run_dir)


def load_report_from_run(run_dir: str | Path) -> dict[str, Any]:
    import pyimgano.workbench.load_run as load_run

    return load_run.load_report_from_run(run_dir)


def select_category_report(
    report: Mapping[str, Any],
    *,
    category: str | None,
) -> tuple[str | None, Mapping[str, Any]]:
    import pyimgano.workbench.load_run as load_run

    return load_run.select_category_report(report, category=category)


def extract_threshold(report_payload: Mapping[str, Any]) -> float | None:
    import pyimgano.workbench.load_run as load_run

    return load_run.extract_threshold(report_payload)


def resolve_checkpoint_path(
    run_dir: str | Path,
    report_payload: Mapping[str, Any],
) -> Path | None:
    import pyimgano.workbench.load_run as load_run

    return load_run.resolve_checkpoint_path(run_dir, report_payload)


def load_checkpoint_into_detector(detector: Any, checkpoint_path: str | Path) -> None:
    import pyimgano.workbench.load_run as load_run

    load_run.load_checkpoint_into_detector(detector, checkpoint_path)


__all__ = [
    "extract_threshold",
    "load_checkpoint_into_detector",
    "load_report_from_run",
    "load_workbench_config_from_run",
    "resolve_checkpoint_path",
    "select_category_report",
]
