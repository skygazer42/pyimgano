from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from pyimgano.reporting.environment import collect_environment
from pyimgano.reporting.report import save_run_report
from pyimgano.reporting.runs import (
    WorkbenchRunPaths,
    build_workbench_run_dir_name,
    build_workbench_run_paths,
    ensure_run_dir,
)
from pyimgano.workbench.config import WorkbenchConfig


@dataclass(frozen=True)
class WorkbenchRunContext:
    run_dir: Path
    paths: WorkbenchRunPaths


def initialize_workbench_run_context(
    *,
    config: WorkbenchConfig,
    recipe_name: str,
) -> WorkbenchRunContext | None:
    if not bool(config.output.save_run):
        return None

    category_for_name = (
        None if str(config.dataset.category).lower() == "all" else str(config.dataset.category)
    )
    name = build_workbench_run_dir_name(
        dataset=str(config.dataset.name),
        recipe=str(recipe_name),
        model=str(config.model.name),
        category=category_for_name,
    )
    run_dir = ensure_run_dir(output_dir=config.output.output_dir, name=name)
    paths = build_workbench_run_paths(run_dir)
    paths.categories_dir.mkdir(parents=True, exist_ok=True)
    paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    paths.artifacts_dir.mkdir(parents=True, exist_ok=True)

    save_run_report(paths.environment_json, collect_environment())
    save_run_report(paths.config_json, {"config": asdict(config)})

    return WorkbenchRunContext(run_dir=run_dir, paths=paths)


__all__ = ["WorkbenchRunContext", "initialize_workbench_run_context"]
